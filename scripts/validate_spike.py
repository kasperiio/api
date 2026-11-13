"""Replay the spike model's median and bands against the base model.

Fits both variants on identical historical rows. Rolling bias and band
calibration use only predictions whose target hours precede each issue time.
No production model or database prediction is changed.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.models  # noqa: F401
import app.forecasting.models  # noqa: F401
from app.database import Base
from app.forecasting import features as F, feedback
from app.forecasting.model import PriceModel


def rolling_calibration(predictions):
    output = []
    for issue, group in predictions.groupby("issued_at", sort=True):
        past = predictions[(predictions.target_ts < issue) & (predictions.issued_at < issue)
                           & (predictions.target_ts >= issue - pd.Timedelta(days=90))]
        correction = feedback.fit_correction(past, issue, source="validation")
        corrected = correction.apply(group.set_index("target_ts")[["p10", "p50", "p90"]], issue)
        group = group.copy()
        for col in ("p10", "p50", "p90"):
            group[col + "_corrected"] = corrected[col + "_corrected"].to_numpy()
        output.append(group)
    return pd.concat(output, ignore_index=True)


def summarize(predictions, corrected=False):
    suffix = "_corrected" if corrected else ""
    actual = predictions.actual.to_numpy()
    low, median, high = [predictions[name + suffix].to_numpy() for name in ("p10", "p50", "p90")]
    error = np.abs(actual - median)
    width = high - low
    score = width + 10 * np.maximum(low - actual, 0) + 10 * np.maximum(actual - high, 0)
    pinball = [np.maximum(q * (actual - value), (q - 1) * (actual - value)).mean()
               for q, value in zip((0.1, 0.5, 0.9), (low, median, high))]
    recent = predictions.target_ts >= predictions.target_ts.max() - pd.Timedelta(days=365)
    return {"n": len(predictions), "mae": float(error.mean()),
            "last365_mae": float(error[recent].mean()),
            "spike_mae": float(error[actual > 100].mean()),
            "coverage": float(((actual >= low) & (actual <= high)).mean()),
            "mean_width": float(width.mean()), "interval_score": float(score.mean()),
            "mean_pinball": float(np.mean(pinball))}


def promotion_checks(base, candidate):
    """Predeclared validation gates; live promotion still uses a fresh holdout."""
    return {"lower_mae": candidate["mae"] < base["mae"],
            "lower_recent_mae": candidate["last365_mae"] < base["last365_mae"],
            "lower_spike_mae": candidate["spike_mae"] < base["spike_mae"],
            "coverage": candidate["coverage"] >= max(0.77, base["coverage"] - 0.01),
            "interval_score": candidate["interval_score"] <= base["interval_score"] * 1.02,
            "width": candidate["mean_width"] <= base["mean_width"] * 1.10}


def validate(db, start, end, retrain_days=28, rounds=700):
    frame, y = F.build_full_frame(db, start, end)
    if frame.empty:
        raise ValueError("No forecast history available")
    target = pd.DatetimeIndex(frame._target_ts)
    issued = pd.DatetimeIndex(frame._issued_at)
    eligible = F.subsample_horizons(frame)
    issues = issued.unique().sort_values()
    issues = issues[issues >= start + pd.Timedelta(days=180)]
    if issues.empty:
        raise ValueError("Insufficient warmup history")
    blocks = ((issues - issues[0]) // pd.Timedelta(days=retrain_days)).astype(int)
    X = frame[F.FEATURE_COLUMNS]

    def block(b):
        dates = issues[blocks == b]
        tr = eligible & (target < F.price_known_until(dates[0]))
        ev = issued.isin(dates)
        model = PriceModel(params={"num_threads": 3}, num_boost_round=rounds).fit(X.loc[tr], y.loc[tr])
        base = model.predict(X.loc[ev])
        model.fit_spike(X.loc[tr], y.loc[tr])
        spike = model.predict(X.loc[ev])
        parts = []
        for name, values in (("baseline", base), ("spike", spike)):
            part = values.reset_index(drop=True)
            part["issued_at"], part["target_ts"] = issued[ev], target[ev]
            part["actual"] = y.loc[ev].to_numpy()
            part["variant"] = name
            parts.append(part)
        print(f"Completed block {b + 1}/{int(blocks.max()) + 1}", flush=True)
        return pd.concat(parts, ignore_index=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        raw = pd.concat(pool.map(block, range(int(blocks.max()) + 1)), ignore_index=True)
    summary, scored = {}, []
    for name, group in raw.groupby("variant"):
        corrected = rolling_calibration(group)
        # The first 90 evaluation days establish an independent calibration history.
        valid = corrected.issued_at >= issues.min() + pd.Timedelta(days=90)
        days = feedback.horizon_day(corrected.issued_at, corrected.target_ts)
        valid &= days.between(1, 7)
        selected = corrected[valid]
        if selected.empty:
            raise ValueError("Insufficient evaluation history after calibration warmup")
        summary[name] = {"raw": summarize(selected), "calibrated": summarize(selected, corrected=True)}
        scored.append(corrected)
    checks = promotion_checks(summary["baseline"]["calibrated"], summary["spike"]["calibrated"])
    summary["checks"] = checks
    summary["eligible"] = all(checks.values())
    summary["config"] = {"start": str(start), "end": str(end), "retrain_days": retrain_days,
                         "rounds": rounds, "training_warmup_days": 180, "calibration_warmup_days": 90}
    return summary, pd.concat(scored, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--retrain-days", type=int, default=28)
    parser.add_argument("--rounds", type=int, default=700)
    args = parser.parse_args()
    if min(args.retrain_days, args.rounds) <= 0:
        parser.error("Retraining cadence and rounds must be positive")
    engine = create_engine(f"sqlite:///{Path(args.db).resolve()}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        metrics, predictions = validate(db, pd.Timestamp(args.start, tz="UTC"),
                                        pd.Timestamp(args.end, tz="UTC"), args.retrain_days, args.rounds)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(output / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
