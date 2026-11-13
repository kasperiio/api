"""
Read/write helpers for forecasting tables.

All frames use tz-aware UTC timestamps. These helpers are the only place that
touches the forecasting tables directly, so sources, the model pipeline and the
scheduler share one storage contract.
"""

import json
from datetime import datetime, timezone
from typing import Iterable, List, Optional

import pandas as pd
from sqlalchemy import and_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from app.forecasting.models import (
    DriverData,
    ForecastSnapshot,
    ModelRecord,
    Prediction,
    SpotPriceRaw,
)


def _utc(ts) -> datetime:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return ts.tz_convert("UTC").to_pydatetime()


_BATCH = 5000


def _bulk_upsert(db: Session, model, rows: List[dict], keys: List[str]) -> None:
    """INSERT ... ON CONFLICT(keys) DO UPDATE in batches (SQLite).

    Only the columns present in ``rows`` are updated on conflict."""
    if not rows:
        return
    table = model.__table__
    update_cols = [c for c in rows[0] if c not in keys]
    for i in range(0, len(rows), _BATCH):
        stmt = sqlite_insert(table)
        stmt = stmt.on_conflict_do_update(
            index_elements=keys,
            set_={c: stmt.excluded[c] for c in update_cols},
        )
        db.execute(stmt, rows[i:i + _BATCH])
    db.commit()


def upsert_spot_prices(db: Session, df: pd.DataFrame) -> int:
    """Insert or update a price frame (timestamp, zone, price_eur_mwh, resolution_minutes)."""
    rows = [{
        "timestamp": _utc(row.timestamp),
        "zone": row.zone,
        "price_eur_mwh": float(row.price_eur_mwh),
        "resolution_minutes": int(getattr(row, "resolution_minutes", 60)),
    } for row in df.itertuples(index=False)]
    _bulk_upsert(db, SpotPriceRaw, rows, ["timestamp", "zone"])
    return len(rows)


def upsert_driver_data(db: Session, df: pd.DataFrame, source: str) -> int:
    """Insert or update a driver frame (timestamp, series, value)."""
    now = datetime.now(timezone.utc)
    rows = [{
        "timestamp": _utc(row.timestamp),
        "series": row.series,
        "value": float(row.value),
        "source": source,
        "updated_at": now,
    } for row in df.dropna(subset=["value"]).itertuples(index=False)]
    _bulk_upsert(db, DriverData, rows, ["timestamp", "series"])
    return len(rows)


def insert_snapshot(db: Session, df: pd.DataFrame, issued_at: datetime) -> int:
    """Store a driver frame as a forecast snapshot captured at ``issued_at``."""
    issued = _utc(issued_at)
    rows = [{
        "issued_at": issued,
        "timestamp": _utc(row.timestamp),
        "series": row.series,
        "value": float(row.value),
    } for row in df.dropna(subset=["value"]).itertuples(index=False)]
    _bulk_upsert(db, ForecastSnapshot, rows, ["issued_at", "timestamp", "series"])
    return len(rows)


def load_spot_prices(
    db: Session, start: datetime, end: datetime, zones: Iterable[str] = ("FI",)
) -> pd.DataFrame:
    """Hourly mean prices, wide frame indexed by UTC hour with one column per zone."""
    stmt = select(SpotPriceRaw).where(
        SpotPriceRaw.timestamp >= _utc(start),
        SpotPriceRaw.timestamp < _utc(end),
        SpotPriceRaw.zone.in_(list(zones)),
    )
    rows = db.execute(stmt).scalars().all()
    if not rows:
        return pd.DataFrame(columns=list(zones), index=pd.DatetimeIndex([], tz="UTC"))
    df = pd.DataFrame(
        [(r.timestamp, r.zone, r.price_eur_mwh) for r in rows],
        columns=["timestamp", "zone", "price"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    return df.pivot_table(index="timestamp", columns="zone", values="price", aggfunc="mean")


def load_drivers(
    db: Session, start: datetime, end: datetime, series: Optional[List[str]] = None
) -> pd.DataFrame:
    """Latest driver values, wide frame indexed by UTC hour with one column per series."""
    conds = [DriverData.timestamp >= _utc(start), DriverData.timestamp < _utc(end)]
    if series:
        conds.append(DriverData.series.in_(series))
    rows = db.execute(select(DriverData).where(and_(*conds))).scalars().all()
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    df = pd.DataFrame(
        [(r.timestamp, r.series, r.value) for r in rows],
        columns=["timestamp", "series", "value"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    return df.pivot_table(index="timestamp", columns="series", values="value", aggfunc="mean")


def load_snapshot_asof(
    db: Session, issued_at: datetime, start: datetime, end: datetime
) -> pd.DataFrame:
    """For each (hour, series) the latest snapshot value captured at or before ``issued_at``.

    Wide frame like ``load_drivers``; empty if no snapshots exist yet.
    """
    stmt = select(ForecastSnapshot).where(
        ForecastSnapshot.issued_at <= _utc(issued_at),
        ForecastSnapshot.timestamp >= _utc(start),
        ForecastSnapshot.timestamp < _utc(end),
    )
    rows = db.execute(stmt).scalars().all()
    if not rows:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))
    df = pd.DataFrame(
        [(r.issued_at, r.timestamp, r.series, r.value) for r in rows],
        columns=["issued_at", "timestamp", "series", "value"],
    )
    df = df.sort_values("issued_at").drop_duplicates(["timestamp", "series"], keep="last")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.floor("h")
    return df.pivot_table(index="timestamp", columns="series", values="value", aggfunc="mean")


def save_predictions(
    db: Session,
    df: pd.DataFrame,
    issued_at: datetime,
    model_version: str,
    is_backtest: bool = False,
) -> int:
    """Store predictions. ``df`` is indexed by UTC target hour with columns
    p10, p50, p90 and optionally p10/p50/p90_corrected. Re-issuing replaces earlier rows
    (keeping an already filled actual)."""
    issued = _utc(issued_at)
    def optional(row, col):
        value = row.get(col)
        return None if value is None or pd.isna(value) else float(value)

    rows = []
    for target, row in df.iterrows():
        target_dt = _utc(target)
        rows.append({
            "issued_at": issued,
            "target_ts": target_dt,
            "horizon_hours": int((target_dt - issued).total_seconds() // 3600),
            "model_version": model_version,
            "p10": float(row["p10"]),
            "p50": float(row["p50"]),
            "p90": float(row["p90"]),
            "p10_corrected": optional(row, "p10_corrected"),
            "p50_corrected": optional(row, "p50_corrected"),
            "p90_corrected": optional(row, "p90_corrected"),
            "is_backtest": is_backtest,
        })
    _bulk_upsert(db, Prediction, rows, ["issued_at", "target_ts", "model_version"])
    return len(rows)


def fill_actuals(db: Session) -> int:
    """Copy known FI prices into predictions that are still missing an actual."""
    pending = db.execute(select(Prediction).where(Prediction.actual.is_(None))).scalars().all()
    if not pending:
        return 0
    start = min(p.target_ts for p in pending)
    end = max(p.target_ts for p in pending) + pd.Timedelta(hours=1)
    prices = load_spot_prices(db, start, end, zones=("FI",))
    if prices.empty:
        return 0
    fi = prices["FI"]
    filled = 0
    for p in pending:
        ts = pd.Timestamp(p.target_ts).tz_convert("UTC")
        if ts in fi.index and not pd.isna(fi[ts]):
            p.actual = float(fi[ts])
            filled += 1
    db.commit()
    return filled


def load_predictions(
    db: Session,
    start: datetime,
    end: datetime,
    scored_only: bool = False,
    include_backtest: bool = True,
) -> pd.DataFrame:
    """Predictions with target_ts in [start, end) as a flat frame."""
    conds = [Prediction.target_ts >= _utc(start), Prediction.target_ts < _utc(end)]
    if scored_only:
        conds.append(Prediction.actual.isnot(None))
    if not include_backtest:
        conds.append(Prediction.is_backtest.is_(False))
    rows = db.execute(select(Prediction).where(and_(*conds))).scalars().all()
    cols = ["issued_at", "target_ts", "horizon_hours", "model_version",
            "p10", "p50", "p90", "p10_corrected", "p50_corrected", "p90_corrected",
            "actual", "is_backtest"]
    df = pd.DataFrame([[getattr(r, c) for c in cols] for r in rows], columns=cols)
    for c in ("issued_at", "target_ts"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def register_model(
    db: Session,
    version: str,
    path: str,
    train_start: datetime,
    train_end: datetime,
    holdout_mae: Optional[float],
    metrics: Optional[dict] = None,
    features: Optional[List[str]] = None,
    activate: bool = False,
) -> ModelRecord:
    record = ModelRecord(
        version=version,
        created_at=datetime.now(timezone.utc),
        train_start=_utc(train_start),
        train_end=_utc(train_end),
        path=path,
        holdout_mae=holdout_mae,
        is_active=False,
        metrics_json=json.dumps(metrics) if metrics else None,
        features_json=json.dumps(features) if features else None,
    )
    db.merge(record)
    db.commit()
    if activate:
        activate_model(db, version)
    return record


def activate_model(db: Session, version: str) -> None:
    db.execute(update(ModelRecord).values(is_active=False))
    db.execute(update(ModelRecord).where(ModelRecord.version == version).values(is_active=True))
    db.commit()


def get_active_model(db: Session) -> Optional[ModelRecord]:
    return db.execute(
        select(ModelRecord).where(ModelRecord.is_active.is_(True))
    ).scalar_one_or_none()
