"""Walk-forward bootstrap on a synthetic DB."""

import os
import time

import pandas as pd
import pytest

from app.forecasting import backtest, store
from app.forecasting.model import PriceModel
from tests.test_features import SYN_START, make_session


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bt")
    db = make_session(tmp, days=200)
    old_dir = backtest.MODEL_DIR
    backtest.MODEL_DIR = str(tmp / "models")
    t0 = time.monotonic()
    try:
        summary = backtest.run_bootstrap(
            db, SYN_START, SYN_START + pd.Timedelta(days=200),
            retrain_every_days=7, min_train_days=90,
        )
    finally:
        backtest.MODEL_DIR = old_dir
    summary["_runtime"] = time.monotonic() - t0
    print("bootstrap runtime %.1fs" % summary["_runtime"])
    yield db, summary
    db.close()


def test_beats_same_hour_last_week(result):
    _, s = result
    o = s["overall"]
    assert o["mae"] < o["mae_baseline_week"]
    assert o["mae"] < o["mae_baseline_day"]
    assert set(s["per_horizon_day"]) >= {"D+1", "D+7"}
    for d in s["per_horizon_day"].values():
        assert d["mae"] < d["mae_baseline_week"]
    # Synthetic train (winter) -> test (summer) shift makes intervals under-cover;
    # only check the metric is sane here (coverage is asserted in test_model).
    assert 0.0 < o["p10_p90_coverage"] <= 1.0


def test_summary_contents(result):
    _, s = result
    assert len(s["feature_importance"]) == 20
    assert s["spike_hours"]["n"] > 0 and s["negative_hours"]["n"] >= 0
    assert s["holdout"]["mae"] is not None
    assert s["eval_issues"] > 100


def test_registered_and_active(result):
    db, s = result
    rec = store.get_active_model(db)
    assert rec is not None and rec.version == s["version"]
    assert rec.holdout_mae == pytest.approx(s["holdout"]["mae"])
    assert os.path.exists(rec.path)
    m = PriceModel.load(rec.path)
    assert m.feature_names


def test_backtest_predictions_stored(result):
    db, s = result
    preds = store.load_predictions(db, SYN_START, SYN_START + pd.Timedelta(days=210))
    assert len(preds) > 0
    assert preds["is_backtest"].all()
    assert (preds["model_version"] == s["version"]).all()
    assert preds["actual"].notna().all()
    assert (preds["p10"] <= preds["p50"]).all() and (preds["p50"] <= preds["p90"]).all()
    # no prediction was issued for a target whose price was already known
    assert (preds["target_ts"] > preds["issued_at"]).all()
