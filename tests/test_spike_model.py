import json

import numpy as np
import pandas as pd
import pytest

from app.forecasting.model import PriceModel, transform


@pytest.fixture
def db(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        yield session
    engine.dispose()


@pytest.fixture(scope="module")
def spike_model():
    rng = np.random.default_rng(6)
    X = pd.DataFrame({"scarcity": rng.uniform(-1, 1, 1600), "wind": rng.normal(size=1600)})
    high = X.scarcity > 0.25
    y = pd.Series(np.where(high, 170 + 25 * X.scarcity, 25 - 10 * X.wind) + rng.normal(0, 5, len(X)))
    model = PriceModel(params={"spike_mixture": True, "num_threads": 3,
                               "min_data_in_leaf": 15}, num_boost_round=60).fit(X, y)
    return model, X, y


def test_spike_predictions_and_explanations_match(spike_model):
    model, X, _ = spike_model
    assert set(model.spike_boosters) == {"classifier", "low", "high"}
    predictions = model.predict(X)
    assert np.isfinite(predictions.to_numpy()).all()
    assert (predictions.p10 <= predictions.p50).all()
    assert (predictions.p50 <= predictions.p90).all()
    contrib = model.contributions(X.iloc[:30])
    np.testing.assert_allclose(contrib.sum(axis=1), transform(predictions.p50.iloc[:30]), atol=1e-8)
    assert "spike_adjustment" in contrib
    assert "spike_adjustment" not in model.feature_importance(X.iloc[:30], top=None)
    assert model.explanation_method == "base_model_shap_plus_spike_adjustment"


def test_spike_roundtrip_and_column_alignment(spike_model, tmp_path):
    model, X, _ = spike_model
    path = str(tmp_path / "spike.model")
    model.save(path)
    saved = json.loads((tmp_path / "spike.model").read_text())
    assert saved["format"] == "PriceModelSpikeMixture"  # older apps must not silently ignore experts
    loaded = PriceModel.load(path)
    pd.testing.assert_frame_equal(model.predict(X), loaded.predict(X[X.columns[::-1]]))
    pd.testing.assert_frame_equal(model.contributions(X.iloc[:10]), loaded.contributions(X.iloc[:10]))
    saved["spike_boosters"].pop("high")
    (tmp_path / "spike.model").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="incomplete spike"):
        PriceModel.load(path)


def test_spike_falls_back_with_too_few_weighted_spikes():
    X = pd.DataFrame({"x": np.arange(500, dtype=float)})
    y = pd.Series(np.where(X.x > 250, 150.0, 20.0))
    weights = pd.Series(np.where(y > 100, 0.0, 1.0))
    model = PriceModel(params={"spike_mixture": True}, num_boost_round=10).fit(X, y, weights)
    assert not model.spike_boosters
    assert model.meta["spike_status"] == "insufficient_class_rows"
    assert np.isfinite(model.predict(X).to_numpy()).all()


def test_saved_spike_model_serves_honest_driver_explanations(spike_model, db, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.database import get_db
    from app.forecasting import store, training
    from app.routers import forecast

    model, X, _ = spike_model
    path = str(tmp_path / "api-spike.model")
    model.save(path)
    issue = pd.Timestamp("2026-11-12T12:00Z")
    targets = pd.date_range(issue + pd.Timedelta(days=1), periods=10, freq="h")
    frame = X.iloc[:10].set_axis(targets)
    store.register_model(db, "spike-api", path, issue - pd.Timedelta(days=365), issue,
                         holdout_mae=10.0, features=list(frame.columns), activate=True)
    store.save_predictions(db, model.predict(frame), issue, "spike-api")
    monkeypatch.setattr(forecast.features, "build_prediction_frame", lambda *a, **kw: frame)
    app = FastAPI()
    app.include_router(forecast.router)
    app.dependency_overrides[get_db] = lambda: db
    training.clear_model_cache()
    try:
        with TestClient(app) as client:
            response = client.get("/forecast/drivers", params={"timestamp": targets[0].isoformat()})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["explanation_method"] == "base_model_shap_plus_spike_adjustment"
        assert body["contribution_unit"] == "asinh_eur_mwh_div_10"
        assert body["p50"] == pytest.approx(model.predict(frame).p50.iloc[0])
        adjustment = next(c for c in body["contributions"] if c["feature"] == "spike_adjustment")
        assert adjustment["value"] is None
    finally:
        training.clear_model_cache()
