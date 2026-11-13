"""
Model lifecycle: champion/challenger retraining, issuing forecasts and bootstrapping.

All prices are EUR/MWh without VAT; all timestamps tz-aware UTC.
"""

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.forecasting import backtest, features, feedback, ingest, store
from app.forecasting.model import PriceModel
from app.forecasting.models import ModelRecord, SpotPriceRaw
from app.logging_config import logger

TRAIN_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
HOLDOUT_DAYS = 28
RECENCY_HALF_LIFE_DAYS = 180.0
# Minimum champion-out-of-sample span for a like-for-like comparison (see retrain()).
MIN_FAIR_EVAL_DAYS = 5
MODEL_DIR = Path("data/models")
HORIZON_DAYS = 7

_train_lock = threading.Lock()
_cache_lock = threading.Lock()
_model_cache = {"version": None, "model": None}


# --------------------------------------------------------------------------- helpers

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def last_price_time(db: Session, zone: str = "FI") -> Optional[datetime]:
    """Timestamp of the latest stored day-ahead price for ``zone``."""
    ts = db.execute(
        select(func.max(SpotPriceRaw.timestamp)).where(SpotPriceRaw.zone == zone)
    ).scalar()
    if ts is None:
        return None
    if ts.tzinfo is None:  # func.max bypasses the TZDateTime result processor
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def latest_model_created(db: Session) -> Optional[datetime]:
    ts = db.execute(select(func.max(ModelRecord.created_at))).scalar()
    if ts is not None and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _target_times(X: pd.DataFrame) -> pd.DatetimeIndex:
    """Target hour of every training row (index may be a MultiIndex of issue/target)."""
    idx = X.index
    if isinstance(idx, pd.MultiIndex):
        names = [n for n in idx.names if n in ("target_ts", "target", "timestamp")]
        level = names[0] if names else idx.nlevels - 1
        idx = idx.get_level_values(level)
    return pd.DatetimeIndex(pd.to_datetime(idx, utc=True))


def recency_weights(X: pd.DataFrame, ref: datetime,
                    half_life_days: float = RECENCY_HALF_LIFE_DAYS) -> pd.Series:
    """Exponential decay weights by target age relative to ``ref`` (1.0 at ref)."""
    age_days = (pd.Timestamp(ref) - _target_times(X)) / pd.Timedelta(days=1)
    age = np.clip(np.asarray(age_days, dtype=float), 0.0, None)
    return pd.Series(0.5 ** (age / half_life_days), index=X.index)


def _mae(y: pd.Series, p50: pd.Series) -> Optional[float]:
    err = (np.asarray(y, dtype=float) - np.asarray(p50, dtype=float))
    err = err[~np.isnan(err)]
    return float(np.abs(err).mean()) if err.size else None


def _mae_by_horizon_day(y: pd.Series, p50: pd.Series, horizon_hours: pd.Series) -> dict:
    days = np.clip(np.ceil(np.asarray(horizon_hours, dtype=float) / 24.0), 1, 7).astype(int)
    err = np.abs(np.asarray(y, dtype=float) - np.asarray(p50, dtype=float))
    out = {}
    for d in range(1, 8):
        mask = days == d
        out[str(d)] = float(np.nanmean(err[mask])) if mask.any() else None
    return out


def _align(X: pd.DataFrame, record: Optional[ModelRecord]) -> pd.DataFrame:
    """Reorder/complete columns to the feature list the model was trained with."""
    if record is not None and record.features_json:
        cols = json.loads(record.features_json)
        if cols:
            return X.reindex(columns=cols)
    return X


def _new_version(now: datetime) -> str:
    return "lgbm-" + now.strftime("%Y%m%dT%H%M%SZ")


# --------------------------------------------------------------------------- model cache

def load_active_model(db: Session) -> Tuple[Optional[ModelRecord], Optional[PriceModel]]:
    """Active registry record and its loaded model, cached until the version changes."""
    record = store.get_active_model(db)
    if record is None:
        return None, None
    with _cache_lock:
        if _model_cache["version"] != record.version:
            _model_cache["model"] = PriceModel.load(record.path)
            _model_cache["version"] = record.version
            logger.info("Loaded forecast model %s", record.version)
        return record, _model_cache["model"]


def clear_model_cache() -> None:
    with _cache_lock:
        _model_cache["version"] = None
        _model_cache["model"] = None


# --------------------------------------------------------------------------- retrain

def retrain(db: Session, reason: str, now: Optional[datetime] = None) -> dict:
    """Train a challenger and promote it if it beats the active champion.

    1. Fit the challenger on [TRAIN_START, holdout_start) where holdout is the
       last HOLDOUT_DAYS before the latest known price, with recency weights.
    2. Score challenger and champion on the holdout. The champion may have been
       trained on part of the holdout, so the comparison uses only target hours
       after the champion's train_end when at least MIN_FAIR_EVAL_DAYS of those
       exist; otherwise the whole holdout (biased in the champion's favour, i.e.
       conservative).
    3. If the challenger wins (strictly lower MAE) or there is no usable champion,
       refit it on all data and register+activate it. A losing challenger is
       registered inactive for the record.

    Never runs concurrently with another retrain/bootstrap.
    """
    if not _train_lock.acquire(blocking=False):
        logger.warning("Retrain (%s) skipped: another training run is in progress", reason)
        return {"status": "skipped", "reason": reason, "detail": "training already running"}
    try:
        return _retrain_locked(db, reason, now)
    finally:
        _train_lock.release()


def _retrain_locked(db: Session, reason: str, now: Optional[datetime]) -> dict:
    now = now or _utcnow()
    last = last_price_time(db)
    if last is None:
        return {"status": "no_data", "reason": reason}
    end = last + timedelta(hours=1)
    holdout_start = end - timedelta(days=HOLDOUT_DAYS)
    logger.info("Retrain (%s): train %s..%s, holdout from %s", reason, TRAIN_START, end, holdout_start)

    X_tr, y_tr, _ = features.build_training_frame(db, TRAIN_START, holdout_start)
    X_ho, y_ho, h_ho = features.build_training_frame(db, holdout_start, end)
    if len(X_tr) == 0 or len(X_ho) == 0:
        return {"status": "no_data", "reason": reason, "train_rows": len(X_tr), "holdout_rows": len(X_ho)}

    challenger = PriceModel().fit(X_tr, y_tr, sample_weight=recency_weights(X_tr, holdout_start))
    ch_pred = challenger.predict(X_ho)
    challenger_mae = _mae(y_ho, ch_pred["p50"])

    champion = store.get_active_model(db)
    champion_mae = None
    eval_scope = "full_holdout"
    challenger_cmp = challenger_mae
    if champion is not None:
        try:
            _, champ_model = load_active_model(db)
            champ_pred = champ_model.predict(_align(X_ho, champion))
            mask = np.ones(len(X_ho), dtype=bool)
            champ_end = pd.Timestamp(champion.train_end)
            targets = _target_times(X_ho)
            fair = np.asarray(targets >= champ_end)
            if fair.any() and (targets[fair].max() - targets[fair].min()) >= pd.Timedelta(days=MIN_FAIR_EVAL_DAYS):
                mask = fair
                eval_scope = "after_champion_train_end"
            champion_mae = _mae(y_ho[mask], champ_pred["p50"][mask])
            challenger_cmp = _mae(y_ho[mask], ch_pred["p50"][mask])
        except Exception as e:  # feature set changed, file missing, ...
            logger.error("Champion %s could not be evaluated: %s", champion.version, e)
            champion_mae = None

    promote = champion is None or champion_mae is None or (
        challenger_cmp is not None and challenger_cmp < champion_mae
    )
    version = _new_version(now)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = str(MODEL_DIR / f"{version}.model")
    metrics = {
        "reason": reason,
        "holdout_start": holdout_start.isoformat(),
        "holdout_mae": challenger_mae,
        "holdout_mae_by_day": _mae_by_horizon_day(y_ho, ch_pred["p50"], h_ho),
        "eval_scope": eval_scope,
        "challenger_eval_mae": challenger_cmp,
        "champion_version": champion.version if champion is not None else None,
        "champion_eval_mae": champion_mae,
        "promoted": bool(promote),
    }

    if promote:
        X_all = pd.concat([X_tr, X_ho])
        y_all = pd.concat([y_tr, y_ho])
        final = PriceModel().fit(X_all, y_all, sample_weight=recency_weights(X_all, end))
        final.save(path)
        train_end = end
    else:
        challenger.save(path)
        train_end = holdout_start
    store.register_model(
        db, version=version, path=path, train_start=TRAIN_START, train_end=train_end,
        holdout_mae=challenger_mae, metrics=metrics, features=list(X_tr.columns),
        activate=promote,
    )
    logger.info(
        "Retrain (%s): challenger %s MAE %.2f vs champion %s -> %s",
        reason, version, challenger_cmp if challenger_cmp is not None else float("nan"),
        champion_mae, "promoted" if promote else "kept champion",
    )
    return {"status": "ok", "version": version, "path": path, "activated": bool(promote), **metrics}


# --------------------------------------------------------------------------- predict

# Weather forecasts cover the whole horizon and drive the D+3..D+7 forecast. The
# model always had them in training, so a forecast without them is meaningless.
REQUIRED_WEATHER = ("wind_speed_100m_forecast_ms", "temperature_forecast_c")
MIN_WEATHER_COVERAGE = 0.8


def _check_weather_coverage(X: pd.DataFrame) -> None:
    for col in REQUIRED_WEATHER:
        coverage = float(X[col].notna().mean()) if col in X.columns and len(X) else 0.0
        if coverage < MIN_WEATHER_COVERAGE:
            raise RuntimeError(
                f"{col} covers only {coverage:.0%} of the forecast hours; "
                "not issuing a forecast (is the forecast snapshot job failing?)"
            )


def predict_and_store(db: Session, issued_at: Optional[datetime] = None) -> pd.DataFrame:
    """Issue a forecast with the active model, bias-correct it and store it.

    Returns the stored frame (UTC target index; p10, p50, p90, p50_corrected).
    Raises RuntimeError if no active model exists.
    """
    issued_at = issued_at or _utcnow().replace(second=0, microsecond=0)
    record, model = load_active_model(db)
    if record is None:
        raise RuntimeError("no active forecast model")
    X = features.build_prediction_frame(db, issued_at, horizon_days=HORIZON_DAYS)
    # Hours whose day-ahead price is already stored are not forecast (the model
    # never trains on them and scoring them would skew bias correction).
    # lead_hours < 0 marks them; it follows the prices actually stored, so a
    # late publication still gets a D+1 forecast.
    X = X[X["lead_hours"] >= 0]
    if len(X) == 0:
        logger.warning("Prediction frame empty for issue time %s", issued_at)
        return pd.DataFrame(columns=["p10", "p50", "p90", "p50_corrected"])
    _check_weather_coverage(X)
    preds = model.predict(_align(X, record))[["p10", "p50", "p90"]]
    preds = feedback.bias_correction(db, now=issued_at).apply(preds, issued_at)
    store.save_predictions(db, preds, issued_at, record.version, is_backtest=False)
    logger.info("Stored %s forecast hours issued %s with model %s", len(preds), issued_at, record.version)
    return preds


# --------------------------------------------------------------------------- bootstrap

def ensure_bootstrapped(db: Session, now: Optional[datetime] = None) -> dict:
    """Backfill history and run the walk-forward bootstrap if no model is active.

    Idempotent: returns immediately when an active model exists; backfill is
    resumable so an interrupted bootstrap continues where it left off.
    """
    if store.get_active_model(db) is not None:
        return {"status": "already_bootstrapped"}
    if not _train_lock.acquire(blocking=False):
        return {"status": "skipped", "detail": "training already running"}
    try:
        now = now or _utcnow()
        logger.info("Forecast bootstrap: backfilling %s..%s", TRAIN_START, now)
        backfill = ingest.backfill(db, TRAIN_START, now)
        logger.info("Forecast bootstrap: backfill done %s; running walk-forward backtest", backfill)
        summary = backtest.run_bootstrap(db, TRAIN_START, now)
    finally:
        _train_lock.release()
    result = {"status": "bootstrapped", "backfill": backfill, "bootstrap": summary}
    if store.get_active_model(db) is None:
        logger.warning("Bootstrap finished without an active model; training one directly")
        result["retrain"] = retrain(db, "bootstrap", now=now)
    return result
