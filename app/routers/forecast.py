"""
Forecast endpoints: 7-day FI price forecast, its accuracy, drivers and system status.

Stored forecasts are EUR/MWh without VAT; the default response unit is c/kWh
including VAT like the electricity endpoints.
"""

import json
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Tuple

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from dateutil.tz import gettz as ZoneInfo

from app.database import get_db
from app.forecasting import feedback, features, scheduler, store, training
from app.forecasting import schemas
from app.forecasting.models import DriverData, Prediction, SpotPriceRaw
from app.logging_config import logger
from app.providers.entsoe import EntsoeClient
from app.routers.electricity import ensure_utc

VAT_RATE = EntsoeClient.VAT_RATE
BIAS_COLUMNS = {"bias", "base_value", "expected_value", "const", "intercept", "_bias"}

router = APIRouter(prefix="/forecast", tags=["forecast"])


class Unit(str, Enum):
    c_kwh_vat = "c_kwh_vat"
    eur_mwh = "eur_mwh"


def to_unit(eur_mwh: float, unit: Unit) -> float:
    """EUR/MWh (excl. VAT) -> requested unit."""
    if unit == Unit.c_kwh_vat:
        return round(eur_mwh / 10.0 * VAT_RATE, 3)
    return round(eur_mwh, 3)


def _tz(timezone_str: str):
    if timezone_str == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(timezone_str)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unknown timezone '{timezone_str}'")


def _aware(ts: Optional[datetime]) -> Optional[datetime]:
    """Aggregates (min/max) bypass TZDateTime, so attach UTC to naive results."""
    if ts is None:
        return None
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts


def latest_issue(db: Session) -> Tuple[Optional[datetime], Optional[str]]:
    """Latest live issue time, preferring the active model, else any model."""
    record = store.get_active_model(db)
    if record is not None:
        ts = db.execute(select(func.max(Prediction.issued_at)).where(
            Prediction.is_backtest.is_(False), Prediction.model_version == record.version,
        )).scalar()
        if ts is not None:
            return _aware(ts), record.version
    row = db.execute(
        select(Prediction.issued_at, Prediction.model_version)
        .where(Prediction.is_backtest.is_(False))
        .order_by(Prediction.issued_at.desc()).limit(1)
    ).first()
    if row is None:
        return None, None
    return _aware(row[0]), row[1]


@router.get("", response_model=schemas.ForecastResponse)
def get_forecast(
    timezone_str: str = Query("UTC", description="Timezone for response timestamps, e.g. 'Europe/Helsinki'"),
    days: int = Query(7, ge=1, le=7, description="Return target hours within this many days of the issue time"),
    unit: Unit = Query(Unit.c_kwh_vat, description="c_kwh_vat (c/kWh incl. VAT) or eur_mwh (EUR/MWh excl. VAT)"),
    db: Session = Depends(get_db),
):
    """Latest issued FI price forecast with p10/p50/p90 bands.

    p50 is the bias-corrected median when available and p10/p90 the calibrated
    band around it (older rows without one fall back to the raw band shifted by
    the p50 correction). ``known_price`` is filled for
    hours whose day-ahead price is already published.
    """
    tz = _tz(timezone_str)
    issued_at, version = latest_issue(db)
    if issued_at is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No forecast has been issued yet. The model may still be bootstrapping; see /forecast/status.",
        )
    end = issued_at + timedelta(days=days)
    df = store.load_predictions(db, issued_at - timedelta(hours=1), end, include_backtest=False)
    df = df[(df["issued_at"] == pd.Timestamp(issued_at)) & (df["model_version"] == version)]
    df = df.sort_values("target_ts")

    known = pd.Series(dtype=float)
    if not df.empty:
        prices = store.load_spot_prices(
            db, df["target_ts"].min(), df["target_ts"].max() + pd.Timedelta(hours=1), zones=("FI",)
        )
        if "FI" in prices.columns:
            known = prices["FI"].dropna()

    points = []
    for row in df.itertuples(index=False):
        p50 = row.p50_corrected if pd.notna(row.p50_corrected) else row.p50
        if pd.notna(row.p10_corrected) and pd.notna(row.p90_corrected):
            p10, p90 = row.p10_corrected, row.p90_corrected
        else:
            shift = p50 - row.p50
            p10, p90 = row.p10 + shift, row.p90 + shift
        p10, p90 = min(p10, p50), max(p90, p50)
        ts = pd.Timestamp(row.target_ts)
        actual = known.get(ts) if len(known) else None
        points.append(schemas.ForecastPoint(
            timestamp=ts.to_pydatetime().astimezone(tz),
            p10=to_unit(p10, unit), p50=to_unit(p50, unit), p90=to_unit(p90, unit),
            known_price=to_unit(float(actual), unit) if actual is not None and pd.notna(actual) else None,
        ))
    return schemas.ForecastResponse(
        issued_at=issued_at.astimezone(tz), model_version=version,
        unit=unit.value, timezone=timezone_str, points=points,
    )


@router.get("/accuracy", response_model=schemas.AccuracyResponse)
def get_accuracy(db: Session = Depends(get_db)):
    """Rolling MAE (EUR/MWh) per horizon day over 14 and 28 days, plus drift status.

    Uses live forecasts once any have been scored, otherwise the bootstrap backtest.
    """
    acc = feedback.compute_accuracy(db)
    drift = feedback.drift_status(db)
    record = store.get_active_model(db)
    return schemas.AccuracyResponse(
        source=acc["source"],
        windows={f"{w}d": rows for w, rows in acc["windows"].items()},
        model_version=record.version if record else None,
        holdout_mae=record.holdout_mae if record else None,
        drift=drift,
    )


@router.get("/drivers", response_model=schemas.DriversResponse)
def get_drivers(
    timestamp: datetime = Query(..., description="Target hour; naive values are interpreted in timezone_str"),
    timezone_str: str = Query("UTC", description="Timezone for naive input and the response"),
    db: Session = Depends(get_db),
):
    """Top-10 base-model SHAP contributions and model adjustments in asinh space; p50 is EUR/MWh."""
    tz = _tz(timezone_str)
    target = pd.Timestamp(ensure_utc(timestamp, timezone_str)).floor("h")
    record, model = training.load_active_model(db)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active forecast model yet.")
    issued_at, _ = latest_issue(db)
    issued_at = issued_at or datetime.now(timezone.utc).replace(second=0, microsecond=0)

    X = features.build_prediction_frame(db, issued_at, horizon_days=training.HORIZON_DAYS)
    X = training._align(X, record)
    idx = pd.DatetimeIndex(pd.to_datetime(X.index, utc=True))
    if target not in idx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{target.isoformat()} is outside the forecast horizon of the issue at {issued_at.isoformat()}.",
        )
    pos = idx.get_loc(target)
    row_X = X.iloc[[pos]]
    contrib = model.contributions(row_X).iloc[0]
    p50 = float(model.predict(row_X)["p50"].iloc[0])

    bias_cols = [c for c in contrib.index if str(c).lower() in BIAS_COLUMNS]
    base_value = float(contrib[bias_cols].sum()) if bias_cols else None
    contrib = contrib.drop(labels=bias_cols).astype(float)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:10]
    items = []
    for feat, val in top.items():
        fv = row_X.iloc[0].get(feat) if feat in row_X.columns else None
        items.append(schemas.DriverContribution(
            feature=str(feat),
            value=float(fv) if fv is not None and pd.notna(fv) else None,
            contribution=round(float(val), 3),
        ))
    return schemas.DriversResponse(
        timestamp=target.to_pydatetime().astimezone(tz), issued_at=issued_at.astimezone(tz),
        model_version=record.version, base_value=base_value, p50=p50, contributions=items,
        explanation_method=getattr(model, "explanation_method", "base_model_shap"),
    )


def _coverage(db: Session, column, *conds) -> schemas.CoverageRange:
    stmt = select(func.min(column), func.max(column), func.count())
    if conds:
        stmt = stmt.where(*conds)
    lo, hi, n = db.execute(stmt).one()
    return schemas.CoverageRange(min=_aware(lo), max=_aware(hi), rows=int(n or 0))


@router.get("/status", response_model=schemas.StatusResponse)
def get_status(db: Session = Depends(get_db)):
    """Scheduler state, active model and stored data coverage."""
    record = store.get_active_model(db)
    model_info = None
    if record is not None:
        metrics = None
        if record.metrics_json:
            try:
                metrics = json.loads(record.metrics_json)
            except ValueError:
                logger.warning("Invalid metrics_json for model %s", record.version)
        model_info = schemas.ModelInfo(
            version=record.version, created_at=record.created_at, train_start=record.train_start,
            train_end=record.train_end, holdout_mae=record.holdout_mae, metrics=metrics,
        )
    coverage = {
        "spot_prices_fi": _coverage(db, SpotPriceRaw.timestamp, SpotPriceRaw.zone == "FI"),
        "spot_prices_all": _coverage(db, SpotPriceRaw.timestamp),
        "driver_data": _coverage(db, DriverData.timestamp),
        "predictions_live": _coverage(db, Prediction.target_ts, Prediction.is_backtest.is_(False)),
        "predictions_backtest": _coverage(db, Prediction.target_ts, Prediction.is_backtest.is_(True)),
    }
    issued_at, _ = latest_issue(db)
    return schemas.StatusResponse(
        scheduler=scheduler.get_status(), active_model=model_info,
        coverage=coverage, latest_issued_at=issued_at,
    )
