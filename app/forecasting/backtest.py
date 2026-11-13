"""Walk-forward bootstrap: replay history to train the first model and seed predictions.

The feature frame for the whole period is built once (training-style: drivers
from history with horizon-dependent noise, see features.py) and then sliced per
walk-forward step, so each retrain only costs a LightGBM fit. Backtest
predictions use the same noisy drivers as training, so the reported MAE is not
flattered by perfect-foresight wind/consumption. Walk-forward models use a
faster config (higher learning rate, fewer rounds) than the final model.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting import store
from app.forecasting.features import (
    FEATURE_COLUMNS,
    HELSINKI,
    build_full_frame,
    price_known_until,
    subsample_horizons,
)
from app.forecasting.model import PriceModel

logger = logging.getLogger(__name__)

# Intended data start: post Russian-import era, OL3 in regular operation from April 2023.
DEFAULT_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
MODEL_DIR = os.path.join("data", "models")
HOLDOUT_DAYS = 28
WALK_FORWARD_PARAMS = {"learning_rate": 0.1, "num_leaves": 31}
WALK_FORWARD_ROUNDS = 250
# LightGBM's per-iteration overhead dominates at these data sizes and extra
# OpenMP threads barely help (many concurrent multi-thread teams even thrash), so
# walk-forward blocks train concurrently (ctypes releases the GIL), each model
# with single-threaded boosters (3 quantiles in parallel -> 3 threads per block).
PARALLEL_BLOCKS = max(1, min(4, (os.cpu_count() or 3) // 3))
MIN_TRAIN_ROWS = 500
IMPORTANCE_SAMPLE = 1000


def _mae(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[m] - b[m]))) if m.any() else None


def _metrics(df: pd.DataFrame) -> Dict:
    """MAE summary for a frame with p10/p50/p90/actual/baseline_* columns."""
    act = df["actual"].to_numpy()
    out: Dict = {"n": int(len(df))}
    out["mae"] = _mae(df["p50"].to_numpy(), act)
    out["mae_baseline_week"] = _mae(df["baseline_week"].to_numpy(), act)
    out["mae_baseline_day"] = _mae(df["baseline_day"].to_numpy(), act)
    inside = (act >= df["p10"].to_numpy()) & (act <= df["p90"].to_numpy())
    out["p10_p90_coverage"] = float(inside.mean()) if len(df) else None
    return out


def _version() -> str:
    return "v" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_bootstrap(
    db: Session,
    start: datetime,
    end: datetime,
    retrain_every_days: int = 7,
    min_train_days: int = 90,
) -> dict:
    """For each day D in [start+min_train_days, end): train on data before D (retraining
    every ``retrain_every_days``), predict D+1..D+7 as issued at D 12:00 Europe/Helsinki,
    store with is_backtest=True, fill actuals. Finally train on everything, register and
    activate it. Returns summary metrics (MAE per horizon day, feature importance).

    Backtest predictions are stored under the final model's version with
    is_backtest=True. Use DEFAULT_START (2023-01-01) as ``start`` unless there is a
    reason not to. Model files go to MODEL_DIR/<version>.model."""
    t0 = time.monotonic()
    start = pd.Timestamp(start).tz_convert("UTC")
    end = pd.Timestamp(end).tz_convert("UTC")
    version = _version()

    logger.info("Bootstrap %s: building features %s .. %s", version, start, end)
    frame, y = build_full_frame(db, start, end, noise=True)
    if frame.empty:
        raise ValueError("no training data in the requested range")
    y_arr = y.to_numpy()
    X = frame[FEATURE_COLUMNS]
    issued = pd.DatetimeIndex(frame["_issued_at"])
    target = pd.DatetimeIndex(frame["_target_ts"])
    train_mask = subsample_horizons(frame)
    logger.info("Features built: %d rows (%d training rows) in %.1fs",
                len(frame), int(train_mask.sum()), time.monotonic() - t0)

    eval_start = start + pd.Timedelta(days=min_train_days)
    eval_issues = issued.unique().sort_values()
    eval_issues = eval_issues[eval_issues >= eval_start]
    if len(eval_issues) == 0:
        raise ValueError("range too short for min_train_days=%d" % min_train_days)

    # Walk forward in blocks of retrain_every_days issue days
    block_id = ((eval_issues - eval_issues[0]) // pd.Timedelta(days=retrain_every_days)).astype(int)
    issued_i8, target_i8 = issued.asi8, target.asi8
    n_blocks = int(block_id.max()) + 1

    def run_block(b: int) -> Optional[pd.DataFrame]:
        block = eval_issues[block_id == b]
        if len(block) == 0:
            return None
        cutoff = price_known_until(block[0])
        tr = train_mask & (target_i8 < cutoff.value)
        if tr.sum() < MIN_TRAIN_ROWS:
            logger.info("Block %d/%d: only %d training rows, skipping", b + 1, n_blocks, int(tr.sum()))
            return None
        tb = time.monotonic()
        params = dict(WALK_FORWARD_PARAMS, num_threads=3)  # 1 per quantile booster
        model = PriceModel(params=params, num_boost_round=WALK_FORWARD_ROUNDS)
        model.fit(X.loc[tr], y.loc[tr])
        ev = np.isin(issued_i8, block.asi8)
        pred = model.predict(X.loc[ev])
        piece = pd.DataFrame({
            "issued_at": issued[ev],
            "target_ts": target[ev],
            "horizon_hours": X.loc[ev, "horizon_hours"].to_numpy(),
            "p10": pred["p10"].to_numpy(),
            "p50": pred["p50"].to_numpy(),
            "p90": pred["p90"].to_numpy(),
            "actual": y_arr[ev],
            "baseline_week": X.loc[ev, "fi_same_hour_lastweek"].to_numpy(),
            "baseline_day": X.loc[ev, "fi_same_hour_lastday"].to_numpy(),
        })
        logger.info("Block %d/%d (%s): trained on %d rows, predicted %d rows in %.1fs",
                    b + 1, n_blocks, block[0].date(), int(tr.sum()), len(piece),
                    time.monotonic() - tb)
        return piece

    with ThreadPoolExecutor(max_workers=PARALLEL_BLOCKS) as pool:
        pieces: List[pd.DataFrame] = [p for p in pool.map(run_block, range(n_blocks)) if p is not None]
    if not pieces:
        raise ValueError("no walk-forward block had enough training data")
    bt = pd.concat(pieces, ignore_index=True)
    t_wf = time.monotonic() - t0

    # Persist backtest predictions
    ts = time.monotonic()
    for issue, grp in bt.groupby("issued_at", sort=True):
        out = grp.set_index("target_ts")[["p10", "p50", "p90"]]
        store.save_predictions(db, out, issue.to_pydatetime(), version, is_backtest=True)
    filled = store.fill_actuals(db)
    logger.info("Stored %d backtest predictions (%d actuals filled) in %.1fs",
                len(bt), filled, time.monotonic() - ts)

    # Metrics
    iss_local = pd.DatetimeIndex(bt["issued_at"]).tz_convert(HELSINKI).tz_localize(None).normalize()
    tgt_local = pd.DatetimeIndex(bt["target_ts"]).tz_convert(HELSINKI).tz_localize(None).normalize()
    bt["horizon_day"] = ((tgt_local - iss_local) // pd.Timedelta(days=1)).astype(int)
    # Compare on rows where every method has a value
    cmp_ = bt.dropna(subset=["actual", "p50", "baseline_week", "baseline_day"])
    per_day = {}
    for d, g in cmp_.groupby("horizon_day"):
        per_day["D+%d" % d] = _metrics(g)
    spike_thr = float(np.nanpercentile(cmp_["actual"], 90))
    holdout_from = pd.Timestamp(bt["issued_at"].max()) - pd.Timedelta(days=HOLDOUT_DAYS)
    holdout = cmp_[cmp_["issued_at"] > holdout_from]
    summary: Dict = {
        "version": version,
        "data_start": start.isoformat(),
        "data_end": end.isoformat(),
        "eval_issues": int(bt["issued_at"].nunique()),
        "overall": _metrics(cmp_),
        "per_horizon_day": per_day,
        "spike_threshold_eur_mwh": spike_thr,
        "spike_hours": _metrics(cmp_[cmp_["actual"] > spike_thr]),
        "negative_hours": _metrics(cmp_[cmp_["actual"] < 0]),
        "holdout_days": HOLDOUT_DAYS,
        "holdout": _metrics(holdout),
    }
    holdout_mae = summary["holdout"]["mae"]

    # Final model on everything
    tf = time.monotonic()
    final = PriceModel().fit(X.loc[train_mask], y.loc[train_mask])
    sample = X.loc[train_mask]
    if len(sample) > IMPORTANCE_SAMPLE:
        sample = sample.sample(IMPORTANCE_SAMPLE, random_state=0)
    importance = final.feature_importance(sample, top=20)
    summary["feature_importance"] = {k: float(v) for k, v in importance.items()}
    path = os.path.join(MODEL_DIR, version + ".model")
    final.save(path)
    logger.info("Final model trained on %d rows in %.1fs -> %s",
                int(train_mask.sum()), time.monotonic() - tf, path)

    summary["train_rows"] = int(train_mask.sum())
    summary["timings_s"] = {
        "walk_forward": round(t_wf, 1),
        "total": round(time.monotonic() - t0, 1),
    }
    store.register_model(
        db,
        version=version,
        path=path,
        train_start=target.min().to_pydatetime(),
        train_end=(target[train_mask].max() + pd.Timedelta(hours=1)).to_pydatetime(),
        holdout_mae=holdout_mae,
        metrics=summary,
        features=FEATURE_COLUMNS,
        activate=True,
    )
    o = summary["overall"]
    logger.info("Bootstrap %s done in %.1fs: MAE %.2f vs week-baseline %.2f / day-baseline %.2f",
                version, summary["timings_s"]["total"], o["mae"] or float("nan"),
                o["mae_baseline_week"] or float("nan"), o["mae_baseline_day"] or float("nan"))
    return summary
