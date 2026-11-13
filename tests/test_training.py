"""Tests for app.forecasting.training with fake features/model/ingest/backtest."""

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.forecasting.models  # noqa: F401
import app.models  # noqa: F401
from app.database import Base
from app.forecasting import feedback, store, training
from app.forecasting.models import ModelRecord

LAST_PRICE = datetime(2026, 11, 1, 22, tzinfo=timezone.utc)
OLD_TRAIN_END = datetime(2026, 9, 1, tzinfo=timezone.utc)


class FakeModel:
    """p50 = x + offset; newly fitted models take ``next_offset``."""

    next_offset = 0.0
    loads = 0
    fits = []

    def __init__(self, offset=None):
        self.offset = offset

    def fit(self, X, y, sample_weight=None):
        assert sample_weight is not None and len(sample_weight) == len(X)
        FakeModel.fits.append((X.index.min(), X.index.max()))
        self.offset = FakeModel.next_offset
        return self

    def predict(self, X):
        p50 = X["x"] + self.offset
        return pd.DataFrame({"p10": p50 - 5, "p50": p50, "p90": p50 + 5}, index=X.index)

    def contributions(self, X):
        return pd.DataFrame({"x": X["x"], "bias": 1.0}, index=X.index)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"offset": self.offset}, f)

    @classmethod
    def load(cls, path):
        cls.loads += 1
        with open(path) as f:
            return cls(json.load(f)["offset"])


def fake_training_frame(db, start, end, noise=True):
    idx = pd.date_range(pd.Timestamp(start).ceil("D"), pd.Timestamp(end), freq="D", inclusive="left")
    X = pd.DataFrame({"x": np.arange(len(idx), dtype=float)}, index=idx)
    y = X["x"].copy()  # a perfect model has offset 0
    h = pd.Series(24, index=idx)
    return X, y, h


def fake_prediction_frame(db, issued_at, horizon_days=7):
    start = pd.Timestamp(issued_at).ceil("h")
    idx = pd.date_range(start, periods=horizon_days * 24, freq="h")
    lead = (idx - training.features.price_known_until(issued_at)) / pd.Timedelta(hours=1)
    return pd.DataFrame({"x": 50.0, "wind_speed_100m_forecast_ms": 8.0, "temperature_forecast_c": 1.0,
                         "lead_hours": lead}, index=idx)


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    monkeypatch.setattr(training, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(training, "PriceModel", FakeModel)
    monkeypatch.setattr(training.features, "build_training_frame", fake_training_frame)
    monkeypatch.setattr(training.features, "build_prediction_frame", fake_prediction_frame)
    FakeModel.next_offset, FakeModel.loads, FakeModel.fits = 0.0, 0, []
    training.clear_model_cache()
    store.upsert_spot_prices(session, pd.DataFrame({
        "timestamp": [pd.Timestamp(LAST_PRICE)], "zone": ["FI"], "price_eur_mwh": [40.0],
    }))
    yield session
    session.close()
    training.clear_model_cache()


def add_champion(db, tmp_path, offset, train_end=OLD_TRAIN_END, version="champ"):
    path = str(tmp_path / f"{version}.model")
    FakeModel(offset).save(path)
    store.register_model(db, version, path, training.TRAIN_START, train_end,
                         holdout_mae=abs(offset), features=["x"], activate=True)


def test_recency_weights_half_life():
    ref = pd.Timestamp("2026-01-01", tz="UTC")
    idx = pd.DatetimeIndex([ref - pd.Timedelta(days=d) for d in (0, 180, 360)])
    w = training.recency_weights(pd.DataFrame({"x": 0}, index=idx), ref)
    assert w.tolist() == pytest.approx([1.0, 0.5, 0.25])


def test_recency_weights_multiindex_uses_target_level():
    idx = pd.MultiIndex.from_arrays([
        pd.DatetimeIndex(["2025-01-01"] * 2, tz="UTC"),
        pd.DatetimeIndex(["2025-07-01", "2025-12-28"], tz="UTC"),
    ], names=["issued_at", "target_ts"])
    w = training.recency_weights(pd.DataFrame({"x": [0, 0]}, index=idx), pd.Timestamp("2025-12-28", tz="UTC"))
    assert w.tolist() == pytest.approx([0.5, 1.0])


def test_first_retrain_activates_without_champion(db):
    FakeModel.next_offset = 3.0
    res = training.retrain(db, "test")
    assert res["status"] == "ok" and res["activated"] is True
    assert res["holdout_mae"] == pytest.approx(3.0)
    active = store.get_active_model(db)
    assert active.version == res["version"]
    assert active.holdout_mae == pytest.approx(3.0)
    assert pd.Timestamp(active.train_end) == pd.Timestamp(LAST_PRICE) + pd.Timedelta(hours=1)
    # holdout fit, then a full refit that also covers the holdout
    assert len(FakeModel.fits) == 2
    assert FakeModel.fits[1][1] > FakeModel.fits[0][1]


def test_better_challenger_is_promoted(db, tmp_path):
    add_champion(db, tmp_path, offset=5.0)
    FakeModel.next_offset = 2.0
    res = training.retrain(db, "weekly")
    assert res["activated"] is True
    assert res["champion_eval_mae"] == pytest.approx(5.0)
    assert res["challenger_eval_mae"] == pytest.approx(2.0)
    assert res["eval_scope"] == "after_champion_train_end"
    assert store.get_active_model(db).version == res["version"]


def test_worse_challenger_is_not_activated(db, tmp_path):
    add_champion(db, tmp_path, offset=5.0)
    FakeModel.next_offset = -8.0
    res = training.retrain(db, "weekly")
    assert res["activated"] is False
    assert store.get_active_model(db).version == "champ"
    rec = db.get(ModelRecord, res["version"])  # registered for the record, inactive
    assert rec is not None and rec.is_active is False
    assert len(FakeModel.fits) == 1  # no full refit for a loser


def test_equal_challenger_is_not_activated(db, tmp_path):
    add_champion(db, tmp_path, offset=5.0)
    FakeModel.next_offset = 5.0
    assert training.retrain(db, "weekly")["activated"] is False


def test_better_point_forecast_with_worse_bands_is_not_promoted(db, tmp_path, monkeypatch):
    add_champion(db, tmp_path, offset=5.0)
    FakeModel.next_offset = 2.0
    original = FakeModel.predict

    def bad_bands(self, X):
        predictions = original(self, X)
        if self.offset == 2.0:
            predictions["p10"] = predictions["p50"] - 100
            predictions["p90"] = predictions["p50"] + 100
        return predictions

    monkeypatch.setattr(FakeModel, "predict", bad_bands)
    result = training.retrain(db, "weekly")
    assert result["challenger_eval_mae"] < result["champion_eval_mae"]
    assert not result["activated"]
    assert not result["band_checks"]["interval_score"]
    assert store.get_active_model(db).version == "champ"


def test_nonfinite_prediction_cannot_be_the_first_model(db, monkeypatch):
    original = FakeModel.predict

    def invalid(self, X):
        predictions = original(self, X)
        predictions.iloc[0, 0] = np.nan
        return predictions

    monkeypatch.setattr(FakeModel, "predict", invalid)
    result = training.retrain(db, "test")
    assert not result["activated"]
    assert not result["band_checks"]["finite_ordered"]
    assert store.get_active_model(db) is None


@pytest.mark.parametrize("experts_fitted", [True, False])
def test_optional_spike_challenger_preserves_validated_family_during_refit(db, monkeypatch, experts_fitted):
    options = []

    class FakeSpike(FakeModel):
        def __init__(self, offset=None, params=None):
            super().__init__(offset)
            enabled = bool((params or {}).get("spike_mixture"))
            options.append(enabled)
            self.spike_boosters = {"classifier": object()} if enabled and experts_fitted else {}

        def predict(self, X, use_spike=True):
            predictions = super().predict(X)
            return predictions if use_spike else predictions + 2.0

    monkeypatch.setattr(training, "PriceModel", FakeSpike)
    result = training.retrain(db, "spike-test", spike_mixture=True)
    assert result["activated"]
    assert options == [True, experts_fitted]
    assert result["model_kind"] == ("spike_mixture" if experts_fitted else "quantile")


def test_worse_spike_experts_fall_back_to_the_base_challenger(db, monkeypatch):
    class FakeSpike(FakeModel):
        def __init__(self, offset=None, params=None):
            super().__init__(offset)
            self.enabled = bool((params or {}).get("spike_mixture"))
            self.spike_boosters = {"classifier": object()} if self.enabled else {}

        def fit(self, X, y, sample_weight=None):
            super().fit(X, y, sample_weight)
            self.offset = 4.0 if self.enabled else 1.0
            return self

        def predict(self, X, use_spike=True):
            predictions = super().predict(X)
            return predictions if use_spike else predictions - 3.0

    monkeypatch.setattr(training, "PriceModel", FakeSpike)
    result = training.retrain(db, "spike-test", spike_mixture=True)
    assert result["activated"] and result["model_kind"] == "quantile"
    assert not result["spike_selection"]["selected"]
    assert result["holdout_mae"] == pytest.approx(1.0)


def test_champion_trained_on_holdout_uses_full_holdout(db, tmp_path):
    add_champion(db, tmp_path, offset=1.0, train_end=LAST_PRICE)
    FakeModel.next_offset = 4.0
    res = training.retrain(db, "weekly")
    assert res["eval_scope"] == "full_holdout"
    assert res["activated"] is False


def test_unusable_champion_is_replaced(db, tmp_path):
    store.register_model(db, "broken", str(tmp_path / "missing.model"), training.TRAIN_START,
                         training.TRAIN_START, holdout_mae=1.0, activate=True)
    FakeModel.next_offset = 4.0
    res = training.retrain(db, "weekly")
    assert res["champion_eval_mae"] is None and res["activated"] is True


def test_retrain_skips_when_locked(db):
    assert training._train_lock.acquire(blocking=False)
    try:
        assert training.retrain(db, "drift")["status"] == "skipped"
    finally:
        training._train_lock.release()


def test_predict_and_store_saves_bias_corrected_p50(db, tmp_path, monkeypatch):
    add_champion(db, tmp_path, offset=2.0)
    table = {(d, h): 3.0 for d in range(1, 8) for h in range(24)}
    monkeypatch.setattr(feedback, "bias_correction", lambda db, now=None: feedback.BiasCorrection(table))
    issued = datetime(2026, 11, 2, 12, 45, tzinfo=timezone.utc)
    preds = training.predict_and_store(db, issued_at=issued)
    # 13:45 Stockholm: prices are published through the end of the next day
    known_until = training.features.price_known_until(issued)
    assert preds.index.min() == known_until
    assert len(preds) == 168 - int((known_until - pd.Timestamp(issued).ceil("h")) / pd.Timedelta("1h"))
    stored = store.load_predictions(db, issued, issued + timedelta(days=8))
    assert len(stored) == len(preds)
    assert (stored["p50"] == 52.0).all()
    assert (stored["p50_corrected"] == 55.0).all()
    assert (~stored["is_backtest"]).all() and (stored["model_version"] == "champ").all()


def test_model_cache_reloads_on_version_change(db, tmp_path):
    add_champion(db, tmp_path, offset=2.0)
    issued = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)
    training.predict_and_store(db, issued_at=issued)
    training.predict_and_store(db, issued_at=issued + timedelta(hours=1))
    assert FakeModel.loads == 1
    add_champion(db, tmp_path, offset=1.0, version="champ2")
    training.predict_and_store(db, issued_at=issued + timedelta(hours=2))
    assert FakeModel.loads == 2


def test_predict_refuses_without_weather_forecasts(db, tmp_path, monkeypatch):
    add_champion(db, tmp_path, offset=2.0)

    def no_weather(db, issued_at, horizon_days=7):
        frame = fake_prediction_frame(db, issued_at, horizon_days)
        frame["wind_speed_100m_forecast_ms"] = float("nan")
        return frame

    monkeypatch.setattr(training.features, "build_prediction_frame", no_weather)
    issued = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match="wind_speed_100m_forecast_ms"):
        training.predict_and_store(db, issued_at=issued)
    assert store.load_predictions(db, issued, issued + timedelta(days=8)).empty


def test_predict_without_model_raises(db):
    with pytest.raises(RuntimeError):
        training.predict_and_store(db)


def test_ensure_bootstrapped_is_idempotent(db, tmp_path, monkeypatch):
    calls = []

    def fake_backfill(db, start, end):
        calls.append("backfill")
        return {"n": 1}

    def fake_bootstrap(db, start, end, **kw):
        calls.append("bootstrap")
        add_champion(db, tmp_path, offset=1.0, version="boot")
        return {"mae": 1.0}

    monkeypatch.setattr(training.ingest, "backfill", fake_backfill)
    monkeypatch.setattr(training.backtest, "run_bootstrap", fake_bootstrap)
    assert training.ensure_bootstrapped(db)["status"] == "bootstrapped"
    assert training.ensure_bootstrapped(db)["status"] == "already_bootstrapped"
    assert calls == ["backfill", "bootstrap"]


def test_ensure_bootstrapped_falls_back_to_retrain(db, monkeypatch):
    monkeypatch.setattr(training.ingest, "backfill", lambda db, s, e: {})
    monkeypatch.setattr(training.backtest, "run_bootstrap", lambda db, s, e, **kw: {})
    FakeModel.next_offset = 1.0
    res = training.ensure_bootstrapped(db)
    assert res["retrain"]["activated"] is True
    assert store.get_active_model(db) is not None
