"""Compare median price models on identical chronological holdouts.

No model is activated and no predictions are written to the database. Outputs per-variant
metrics and paired predictions for inspection. Example:
python scripts/forecast_experiments.py --db data/electricity_prices.db --out experiments
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.models  # noqa: F401
import app.forecasting.models  # noqa: F401
from app.database import Base
from app.forecasting import features as F, store
from app.forecasting.model import DEFAULT_PARAMS, DEFAULT_ROUNDS, transform, inverse_transform


def compare(db, start, end, retrain_days=14, min_train_days=180, experiment="models"):
    frame, y = F.build_full_frame(db, start, end)
    if frame.empty:
        raise ValueError("No historical prices in the requested interval")
    columns = list(F.FEATURE_COLUMNS)
    simulated = None
    if experiment == "sources":
        from unittest.mock import patch
        with patch.object(store, "load_archived_weather", return_value=pd.DataFrame()):
            simulated, _ = F.build_full_frame(db, start, end)
        if not frame[["_issued_at", "_target_ts"]].equals(simulated[["_issued_at", "_target_ts"]]):
            raise ValueError("Source comparison frames have different target grids")
    issued = pd.DatetimeIndex(frame._issued_at)
    target = pd.DatetimeIndex(frame._target_ts)
    labels = y.to_numpy()
    neighbours = store.load_spot_prices(db, start, end, zones=("SE1", "EE")).reindex(target)
    regime = (np.abs(labels - neighbours.EE.to_numpy()) <
              np.abs(labels - neighbours.SE1.to_numpy())).astype(float)
    regime[neighbours.isna().any(axis=1).to_numpy()] = np.nan
    eligible = F.subsample_horizons(frame)
    issues = issued.unique().sort_values()
    issues = issues[issues >= start + pd.Timedelta(days=min_train_days)]
    if issues.empty:
        raise ValueError("Not enough history for the training warmup")
    blocks = ((issues - issues[0]) // pd.Timedelta(days=retrain_days)).astype(int)
    params = dict(DEFAULT_PARAMS, alpha=0.5, num_threads=2)

    def block(b):
        dates = issues[blocks == b]
        # All training labels end before prices would be published at issue.
        tr = eligible & (target < F.price_known_until(dates[0]))
        ev = issued.isin(dates)
        X = frame.loc[tr, columns]
        Y = labels[tr]
        test = frame.loc[ev, columns]

        if experiment == "sources":
            original_columns = [c for c in columns if c not in F.OUTAGE_FEATURES
                                and c != "eua_auction_eur_t" and c not in F.S.TRANSFER_CAPACITY_SERIES]
            out = pd.DataFrame({"issued_at": issued[ev], "target_ts": target[ev], "actual": labels[ev]})
            for name, train_frame, selected in (
                ("baseline", simulated, original_columns),
                ("archived_weather", frame, original_columns),
                ("all_sources", frame, columns),
            ):
                booster = lgb.train(params, lgb.Dataset(train_frame.loc[tr, selected], transform(Y)),
                                    num_boost_round=DEFAULT_ROUNDS)
                out[name] = inverse_transform(booster.predict(test[selected], num_threads=2))
            return out

        def fit_predict(mask=None, objective="quantile", target_values=None):
            mask = np.ones(len(X), dtype=bool) if mask is None else mask
            values = transform(Y) if target_values is None else target_values
            p = dict(params, objective=objective)
            booster = lgb.train(p, lgb.Dataset(X.loc[mask], values[mask]),
                                num_boost_round=DEFAULT_ROUNDS if objective == "quantile" else 300)
            result = booster.predict(test, num_threads=2)
            return inverse_transform(result) if objective == "quantile" else result

        base = fit_predict()
        out = pd.DataFrame({"issued_at": issued[ev], "target_ts": target[ev],
                            "actual": labels[ev], "baseline": base})
        for name, classes in (("spike", (Y > 100).astype(float)), ("regime", regime[tr])):
            valid = np.isfinite(classes)
            low, high = valid & (classes == 0), valid & (classes == 1)
            if low.sum() < 200 or high.sum() < 200:
                out[name] = base
                continue
            probability = fit_predict(valid, "binary", classes)
            out[name] = ((1 - probability) * fit_predict(low) + probability * fit_predict(high))
        return out

    with ThreadPoolExecutor(max_workers=3) as pool:
        predictions = pd.concat(pool.map(block, range(int(blocks.max()) + 1)), ignore_index=True)
    metrics = {}
    recent = predictions.target_ts >= end - pd.Timedelta(days=365)
    variants = ("baseline", "archived_weather", "all_sources") if experiment == "sources" else ("baseline", "spike", "regime")
    for name in variants:
        err = (predictions.actual - predictions[name]).abs()
        metrics[name] = {"n": len(err), "mae": float(err.mean()),
                         "last365_mae": float(err[recent].mean()),
                         "spike_mae": float(err[predictions.actual > 100].mean())}
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--retrain-days", type=int, default=14)
    parser.add_argument("--experiment", choices=("models", "sources"), default="models")
    args = parser.parse_args()
    if args.retrain_days <= 0:
        parser.error("--retrain-days must be positive")
    engine = create_engine(f"sqlite:///{Path(args.db).resolve()}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        metrics, predictions = compare(db, pd.Timestamp(args.start, tz="UTC"),
                                       pd.Timestamp(args.end, tz="UTC"), args.retrain_days,
                                       experiment=args.experiment)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    predictions.to_csv(out / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
