"""
Feedback loop: score stored predictions against actual prices, learn a residual
bias correction, and detect model drift.

All prices are EUR/MWh without VAT.

Horizon day
    The number of Europe/Helsinki calendar days between the issue time and the
    target hour (target local date - issue local date). A forecast issued on day
    D at 14:45 has horizon day 1 for all hours of D+1, 2 for D+2 and so on. Rows
    for the rest of the issue day (0) or beyond day 7 are excluded from accuracy
    stats; for bias lookups they are clipped into 1..7.

Bias correction and quantiles
    p50 gets an additive shift per (horizon day, hour) bucket. p10 and p90 are
    calibrated separately per horizon day with conformalized quantile
    regression in the model's asinh space: each bound moves outward (or inward)
    by the conformal quantile of how far recent actuals fell beyond it, so about
    10% of actuals land below p10 and 10% above p90. Working in asinh space
    makes the margin grow with the price level, which matters for spikes.
    Quantile GBMs under-cover out of sample, so this mostly widens the band.
    The band then moves with the p50 correction.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting import store
from app.forecasting.model import inverse_transform, transform
from app.logging_config import logger

LOCAL_TZ = "Europe/Helsinki"
HORIZON_DAYS = tuple(range(1, 8))
ACCURACY_WINDOWS = (14, 28)

# Tuned on the walk-forward backtest: short windows and light shrinkage chase
# noise and made both MAE and band coverage worse.
BIAS_LOOKBACK_DAYS = 90
BIAS_SHRINK_K = 500.0       # weight = n / (n + K)
BIAS_CLIP_EUR_MWH = 15.0    # absolute cap on a bucket offset

BAND_LOOKBACK_DAYS = 90     # 28 days under-covers (~71%); 90 reaches ~80%
BAND_TAIL = 0.1             # target miss rate below p10 and above p90
BAND_SHRINK_K = 48.0        # margin = raw * n / (n + K)
BAND_MARGIN_LIMITS = (-1.0, 3.0)  # asinh units; 3.0 is roughly x20 at high prices

DRIFT_WINDOW_DAYS = 14
DRIFT_RATIO = 1.25
DRIFT_MIN_SAMPLES = 7 * 24  # at least a week's worth of scored live hours


def _now(now: Optional[datetime]) -> pd.Timestamp:
    ts = pd.Timestamp(now) if now is not None else pd.Timestamp(datetime.now(timezone.utc))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def horizon_day(issued_at: pd.Series, target_ts: pd.Series) -> pd.Series:
    """Helsinki calendar-day difference between target and issue time (unclipped)."""
    issued_local = pd.to_datetime(issued_at, utc=True).dt.tz_convert(LOCAL_TZ).dt.normalize()
    target_local = pd.to_datetime(target_ts, utc=True).dt.tz_convert(LOCAL_TZ).dt.normalize()
    # normalize() keeps tz; difference across DST is 23/25 h, so round to days
    return ((target_local - issued_local) / pd.Timedelta(days=1)).round().astype(int)


def local_hour(ts) -> pd.Series:
    return pd.to_datetime(pd.Series(ts), utc=True).dt.tz_convert(LOCAL_TZ).dt.hour


def _served_p50(df: pd.DataFrame) -> pd.Series:
    return df["p50_corrected"].where(df["p50_corrected"].notna(), df["p50"]).astype(float)


def _scored(db: Session, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, Optional[str]]:
    """Scored predictions in [start, end): live ones if any exist, else backtest ones."""
    df = store.load_predictions(db, start, end, scored_only=True, include_backtest=True)
    if df.empty:
        return df, None
    live = df[~df["is_backtest"].astype(bool)]
    if not live.empty:
        return live.copy(), "live"
    return df[df["is_backtest"].astype(bool)].copy(), "backtest"


# --------------------------------------------------------------------------- accuracy

def compute_accuracy(db: Session, now: Optional[datetime] = None) -> dict:
    """Rolling MAE per horizon day over the last 14 and 28 days.

    Uses live predictions when any scored live prediction exists in the longest
    window, otherwise backtest predictions. ``mae`` scores the served forecast
    (p50_corrected where available), ``mae_raw`` the raw model p50.
    """
    end = _now(now)
    longest = max(ACCURACY_WINDOWS)
    df, source = _scored(db, end - pd.Timedelta(days=longest), end)
    result = {"source": source, "windows": {}}
    if not df.empty:
        df["horizon_day"] = horizon_day(df["issued_at"], df["target_ts"])
        df = df[df["horizon_day"].between(1, 7)]
        df["err"] = (df["actual"] - _served_p50(df)).abs()
        df["err_raw"] = (df["actual"] - df["p50"]).abs()
    for window in ACCURACY_WINDOWS:
        rows = []
        sub = df[df["target_ts"] >= end - pd.Timedelta(days=window)] if not df.empty else df
        for day in HORIZON_DAYS:
            part = sub[sub["horizon_day"] == day] if not sub.empty else sub
            n = int(len(part))
            rows.append({
                "horizon_day": day,
                "mae": float(part["err"].mean()) if n else None,
                "mae_raw": float(part["err_raw"].mean()) if n else None,
                "n": n,
            })
        result["windows"][window] = rows
    return result


def score(db: Session, now: Optional[datetime] = None) -> dict:
    """Fill actual prices into stored predictions, then return rolling accuracy."""
    filled = store.fill_actuals(db)
    accuracy = compute_accuracy(db, now=now)
    accuracy["filled"] = filled
    logger.info("Forecast scoring: filled %s actuals (source=%s)", filled, accuracy["source"])
    return accuracy


# --------------------------------------------------------------------------- bias

class BiasCorrection:
    """Additive p50 offsets (EUR/MWh) per (horizon day, Helsinki hour) bucket and
    p10/p90 conformal margins (asinh units) per horizon day."""

    def __init__(self, offsets: Optional[Dict[Tuple[int, int], float]] = None,
                 counts: Optional[Dict[Tuple[int, int], int]] = None,
                 source: Optional[str] = None,
                 band_margins: Optional[Dict[int, Tuple[float, float]]] = None):
        self.offsets = offsets or {}
        self.counts = counts or {}
        self.source = source
        self.band_margins = band_margins or {}

    def offset(self, day: int, hour: int) -> float:
        day = int(min(max(day, 1), 7))
        return float(self.offsets.get((day, int(hour)), 0.0))

    def band_margin(self, day: int) -> Tuple[float, float]:
        """(lower, upper) margins in asinh units; positive widens the band."""
        return self.band_margins.get(int(min(max(day, 1), 7)), (0.0, 0.0))

    def offsets_for(self, index: pd.DatetimeIndex, issued_at: datetime) -> pd.Series:
        """Offsets for target hours ``index`` of a forecast issued at ``issued_at``."""
        targets = pd.Series(pd.to_datetime(index, utc=True))
        issued = pd.Series([pd.Timestamp(issued_at)] * len(targets))
        days = horizon_day(issued, targets).clip(1, 7)
        hours = local_hour(targets)
        values = [self.offset(d, h) for d, h in zip(days, hours)]
        return pd.Series(values, index=index, dtype=float)

    def apply(self, preds: pd.DataFrame, issued_at: datetime) -> pd.DataFrame:
        """Return a copy of ``preds`` (p10/p50/p90, UTC target index) with
        p10_corrected, p50_corrected and p90_corrected (monotone)."""
        out = preds.copy()
        shift = self.offsets_for(out.index, issued_at).to_numpy()
        targets = pd.Series(pd.to_datetime(out.index, utc=True))
        days = horizon_day(pd.Series([pd.Timestamp(issued_at)] * len(targets)), targets).clip(1, 7)
        margins = np.array([self.band_margin(d) for d in days], dtype=float).reshape(-1, 2)
        p50 = out["p50"].to_numpy() + shift
        p10 = inverse_transform(transform(out["p10"]) - margins[:, 0]) + shift
        p90 = inverse_transform(transform(out["p90"]) + margins[:, 1]) + shift
        out["p50_corrected"] = p50
        out["p10_corrected"] = np.minimum(p10, p50)
        out["p90_corrected"] = np.maximum(p90, p50)
        return out


def band_margins(df: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    """Per horizon day, conformal (lower, upper) margins in asinh units for the
    scored rows in ``df`` (columns issued_at, target_ts, p10, p90, actual).

    Upper score = T(actual) - T(p90), lower score = T(p10) - T(actual); each
    margin is the finite-sample (1 - BAND_TAIL) quantile of its score, shrunk
    toward 0 for small samples and clipped to BAND_MARGIN_LIMITS.
    """
    if df.empty:
        return {}
    df = df.copy()
    df["horizon_day"] = horizon_day(df["issued_at"], df["target_ts"]).clip(1, 7)
    actual = transform(df["actual"])
    df["upper"] = actual - transform(df["p90"])
    df["lower"] = transform(df["p10"]) - actual
    margins = {}
    for day, grp in df.groupby("horizon_day"):
        n = len(grp)
        level = min(1.0, np.ceil((n + 1) * (1 - BAND_TAIL)) / n)
        weight = n / (n + BAND_SHRINK_K)
        lower, upper = (
            float(np.clip(np.quantile(grp[col], level) * weight, *BAND_MARGIN_LIMITS))
            for col in ("lower", "upper")
        )
        margins[int(day)] = (lower, upper)
    return margins


def fit_correction(scored: pd.DataFrame, end: pd.Timestamp, source: Optional[str] = None) -> BiasCorrection:
    """Fit p50 offsets and band margins from scored predictions with targets before ``end``.

    p50 offset per (horizon day, local hour) bucket over the last BIAS_LOOKBACK_DAYS:
    clip(median(actual - p50) * n / (n + K), -C, C) with K = BIAS_SHRINK_K and
    C = BIAS_CLIP_EUR_MWH. Residuals are taken against the raw p50 so the
    correction does not feed back on itself. Band margins use the last
    BAND_LOOKBACK_DAYS (see ``band_margins``).
    """
    if scored.empty:
        return BiasCorrection(source=None)
    band_df = scored[scored["target_ts"] >= end - pd.Timedelta(days=BAND_LOOKBACK_DAYS)]
    margins = band_margins(band_df)
    df = band_df[band_df["target_ts"] >= end - pd.Timedelta(days=BIAS_LOOKBACK_DAYS)].copy()
    if df.empty:
        return BiasCorrection(source=None, band_margins=margins)
    df["horizon_day"] = horizon_day(df["issued_at"], df["target_ts"]).clip(1, 7)
    df["hour"] = local_hour(df["target_ts"]).values
    df["resid"] = df["actual"].astype(float) - df["p50"].astype(float)
    offsets, counts = {}, {}
    for (day, hour), grp in df.groupby(["horizon_day", "hour"]):
        n = len(grp)
        shrunk = float(np.median(grp["resid"])) * n / (n + BIAS_SHRINK_K)
        offsets[(int(day), int(hour))] = float(np.clip(shrunk, -BIAS_CLIP_EUR_MWH, BIAS_CLIP_EUR_MWH))
        counts[(int(day), int(hour))] = n
    return BiasCorrection(offsets, counts, source, band_margins=margins)


def bias_correction(db: Session, now: Optional[datetime] = None) -> BiasCorrection:
    """``fit_correction`` on all scored predictions (backtest and live) of the
    lookback window, so the calibration doesn't collapse to a handful of rows
    when the first live forecasts get scored."""
    end = _now(now)
    start = end - pd.Timedelta(days=max(BAND_LOOKBACK_DAYS, BIAS_LOOKBACK_DAYS))
    scored = store.load_predictions(db, start, end, scored_only=True, include_backtest=True)
    if scored.empty:
        return BiasCorrection(source=None)
    is_live = ~scored["is_backtest"].astype(bool)
    source = "live" if is_live.all() else ("mixed" if is_live.any() else "backtest")
    return fit_correction(scored, end, source)


# --------------------------------------------------------------------------- drift

def drift_status(db: Session, now: Optional[datetime] = None) -> dict:
    """Compare live 14-day MAE (raw p50, horizon days 1..7) to the active model's holdout MAE."""
    end = _now(now)
    record = store.get_active_model(db)
    holdout = record.holdout_mae if record is not None else None
    df = store.load_predictions(
        db, end - pd.Timedelta(days=DRIFT_WINDOW_DAYS), end, scored_only=True, include_backtest=False
    )
    if not df.empty:
        df = df[horizon_day(df["issued_at"], df["target_ts"]).between(1, 7)]
    n = int(len(df))
    live_mae = float((df["actual"] - df["p50"]).abs().mean()) if n else None
    threshold = holdout * DRIFT_RATIO if holdout else None
    drift = bool(
        threshold is not None and live_mae is not None
        and n >= DRIFT_MIN_SAMPLES and live_mae > threshold
    )
    return {
        "drift": drift,
        "live_mae_14d": live_mae,
        "n": n,
        "min_samples": DRIFT_MIN_SAMPLES,
        "holdout_mae": holdout,
        "threshold": threshold,
        "model_version": record.version if record is not None else None,
    }


def detect_drift(db: Session, now: Optional[datetime] = None) -> bool:
    return drift_status(db, now=now)["drift"]
