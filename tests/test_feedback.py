"""Tests for app.forecasting.feedback: scoring, bias correction and drift."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.forecasting.models  # noqa: F401
import app.models  # noqa: F401
from app.database import Base
from app.forecasting import feedback, store

HEL = "Europe/Helsinki"
NOW = datetime(2026, 11, 10, 12, tzinfo=timezone.utc)


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def add_day(db, issue_day, resid, *, horizon_day=1, hours=range(24), is_backtest=False,
            version="m1", p50=50.0, corrected=None):
    """One forecast issued at issue_day 12:00 Helsinki for the given local hours of
    issue_day + horizon_day, with actual = p50 + resid."""
    base = pd.Timestamp(issue_day).normalize()  # naive local wall clock, DST-safe
    issued = (base + pd.Timedelta(hours=12)).tz_localize(HEL)
    targets = pd.DatetimeIndex([
        (base + pd.Timedelta(days=horizon_day, hours=h)).tz_localize(HEL) for h in hours
    ]).tz_convert("UTC")
    df = pd.DataFrame({"p10": p50 - 10, "p50": p50, "p90": p50 + 10}, index=targets)
    if corrected is not None:
        df["p50_corrected"] = corrected
    store.save_predictions(db, df, issued, version, is_backtest=is_backtest)
    store.upsert_spot_prices(db, pd.DataFrame({
        "timestamp": targets, "zone": "FI", "price_eur_mwh": p50 + resid, "resolution_minutes": 60,
    }))


def test_horizon_day_uses_helsinki_calendar_days():
    issued = pd.Series([pd.Timestamp("2026-11-09 12:45", tz="UTC")] * 3)  # 14:45 local
    targets = pd.Series(pd.to_datetime(
        ["2026-11-09 21:00", "2026-11-09 22:00", "2026-11-16 21:00"], utc=True))
    # 23:00 local same day -> 0, 00:00 local next day -> 1, a week later -> 7
    assert feedback.horizon_day(issued, targets).tolist() == [0, 1, 7]


def test_score_fills_actuals_and_reports_mae_per_horizon_day(db):
    for d in range(5):
        # distinct local hours so the two horizons never share a target hour
        add_day(db, f"2026-11-0{d + 1}", resid=4.0, horizon_day=1, hours=range(12))
        add_day(db, f"2026-11-0{d + 1}", resid=-8.0, horizon_day=3, hours=range(12, 18))
    res = feedback.score(db, now=NOW)
    assert res["filled"] > 0
    assert res["source"] == "live"
    by_day = {r["horizon_day"]: r for r in res["windows"][14]}
    assert by_day[1]["mae"] == pytest.approx(4.0)
    assert by_day[3]["mae"] == pytest.approx(8.0)
    assert by_day[2]["n"] == 0 and by_day[2]["mae"] is None


def test_accuracy_falls_back_to_backtest_and_scores_corrected_p50(db):
    add_day(db, "2026-11-05", resid=10.0, is_backtest=True, corrected=56.0)
    store.fill_actuals(db)
    res = feedback.compute_accuracy(db, now=NOW)
    assert res["source"] == "backtest"
    day1 = res["windows"][28][0]
    assert day1["mae"] == pytest.approx(4.0)       # actual 60 vs corrected 56
    assert day1["mae_raw"] == pytest.approx(10.0)  # actual 60 vs raw 50


def test_bias_correction_shrinks_by_sample_count(db):
    for d in range(10):  # 10 samples per (day 1, hour) bucket, residual +10
        add_day(db, pd.Timestamp("2026-10-25") + pd.Timedelta(days=d), resid=10.0)
    store.fill_actuals(db)
    bias = feedback.bias_correction(db, now=NOW)
    assert bias.counts[(1, 5)] == 10
    assert bias.offset(1, 5) == pytest.approx(10.0 * 10 / (10 + feedback.BIAS_SHRINK_K))
    assert bias.offset(2, 5) == 0.0  # unseen bucket -> no correction


def test_bias_correction_is_clipped_and_uses_median(db):
    for d in range(18):
        add_day(db, pd.Timestamp("2026-10-21") + pd.Timedelta(days=d), resid=-1000.0, hours=[8])
    add_day(db, "2026-11-09", resid=500.0, hours=[8], horizon_day=0)  # day 0 -> clipped into bucket 1
    store.fill_actuals(db)
    bias = feedback.bias_correction(db, now=NOW)
    assert bias.counts[(1, 8)] == 19
    assert bias.offset(1, 8) == -feedback.BIAS_CLIP_EUR_MWH


def test_bias_uses_backtest_and_live_together(db):
    for d in range(5):
        add_day(db, pd.Timestamp("2026-11-01") + pd.Timedelta(days=d), resid=10.0, is_backtest=True)
    add_day(db, "2026-11-07", resid=-5.0, version="m2")
    store.fill_actuals(db)
    bias = feedback.bias_correction(db, now=NOW)
    assert bias.source == "mixed"
    assert bias.counts[(1, 0)] == 6
    assert bias.offset(1, 0) == pytest.approx(10.0 * 6 / (6 + feedback.BIAS_SHRINK_K))


def test_apply_sets_p50_corrected_by_bucket():
    bias = feedback.BiasCorrection({(1, 0): 5.0, (2, 0): -3.0})
    issued = pd.Timestamp("2026-11-09 12:45", tz="UTC")
    idx = pd.DatetimeIndex(pd.to_datetime(["2026-11-09 22:00", "2026-11-10 22:00", "2026-11-09 23:00"], utc=True))
    preds = pd.DataFrame({"p10": 0.0, "p50": 10.0, "p90": 20.0}, index=idx)
    out = bias.apply(preds, issued)
    assert out["p50_corrected"].tolist() == [15.0, 7.0, 10.0]
    assert "p50_corrected" not in preds


def _activate(db, holdout):
    store.register_model(db, "m1", "x.model", NOW, NOW, holdout_mae=holdout, activate=True)


@pytest.mark.parametrize("resid,expected", [(13.0, True), (12.0, False)])
def test_drift_threshold(db, resid, expected):
    _activate(db, 10.0)  # threshold 12.5
    for d in range(8):
        add_day(db, pd.Timestamp("2026-11-01") + pd.Timedelta(days=d), resid=resid)
    store.fill_actuals(db)
    st = feedback.drift_status(db, now=NOW)
    assert st["n"] == 8 * 24
    assert st["threshold"] == pytest.approx(12.5)
    assert st["drift"] is expected
    assert feedback.detect_drift(db, now=NOW) is expected


def test_drift_requires_min_samples_and_ignores_backtest(db):
    _activate(db, 10.0)
    for d in range(3):
        add_day(db, pd.Timestamp("2026-11-01") + pd.Timedelta(days=d), resid=100.0)
    for d in range(10):
        add_day(db, pd.Timestamp("2026-10-28") + pd.Timedelta(days=d), resid=100.0,
                is_backtest=True, version="bt")
    store.fill_actuals(db)
    st = feedback.drift_status(db, now=NOW)
    assert st["n"] == 3 * 24
    assert st["drift"] is False


def test_drift_false_without_active_model(db):
    for d in range(10):
        add_day(db, pd.Timestamp("2026-11-01") + pd.Timedelta(days=d), resid=100.0)
    store.fill_actuals(db)
    assert feedback.detect_drift(db, now=NOW) is False


def _band_frame(actual, p10=40.0, p50=50.0, p90=60.0):
    targets = pd.date_range("2026-10-02 00:00", periods=len(actual), freq="h", tz="UTC")
    return pd.DataFrame({
        "issued_at": targets - pd.Timedelta(days=1), "target_ts": targets,  # all horizon day 1
        "p10": p10, "p50": p50, "p90": p90, "actual": actual,
    })


def test_band_margins_widen_undercovering_band_asymmetrically():
    rng = np.random.default_rng(0)
    # actuals spike far above p90 but rarely drop below p10
    actual = np.where(rng.random(2000) < 0.3, 150.0, 50.0) + rng.normal(0, 2, 2000)
    margins = feedback.band_margins(_band_frame(actual))
    lower, upper = margins[1]
    # spikes sit asinh(15) - asinh(6) ~ 0.91 above p90; lower side needs no widening
    assert 0.85 < upper < 0.95 and lower < 0

    issued = pd.Timestamp("2026-10-01 10:00", tz="UTC")
    idx = pd.date_range("2026-10-02 00:00", periods=20, freq="h", tz="UTC")  # Helsinki day 1
    preds = pd.DataFrame({"p10": 40.0, "p50": 50.0, "p90": 60.0}, index=idx)
    out = feedback.BiasCorrection(band_margins=margins).apply(preds, issued)
    assert (out["p90_corrected"] >= 140).all()
    assert (out["p10_corrected"] <= out["p50_corrected"]).all()
    assert (out["p50_corrected"] <= out["p90_corrected"]).all()


def test_band_margins_grow_with_price_level():
    # the same asinh margin is a wider band in EUR/MWh at high prices
    bias = feedback.BiasCorrection(band_margins={1: (0.5, 0.5)})
    issued = pd.Timestamp("2026-10-01 10:00", tz="UTC")
    idx = pd.date_range("2026-10-02 00:00", periods=2, freq="h", tz="UTC")
    preds = pd.DataFrame({"p10": [10.0, 180.0], "p50": [20.0, 200.0], "p90": [30.0, 220.0]}, index=idx)
    out = bias.apply(preds, issued)
    widths = out["p90_corrected"] - out["p10_corrected"]
    assert widths.iloc[1] > 5 * widths.iloc[0]


def test_band_margins_shrink_toward_zero_with_few_samples():
    few = feedback.band_margins(_band_frame(np.full(5, 150.0)))[1][1]
    many = feedback.band_margins(_band_frame(np.full(500, 150.0)))[1][1]
    assert 0 < few < many / 5
