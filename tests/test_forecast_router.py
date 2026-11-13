"""Tests for the /forecast endpoints (scheduler disabled, temp SQLite)."""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.forecasting.models  # noqa: F401
import app.models  # noqa: F401
from app.database import Base, get_db
from app.forecasting import scheduler, store, training
from app.routers import forecast

ISSUED = datetime(2026, 11, 12, 12, 45, tzinfo=timezone.utc)  # 14:45 Helsinki
FIRST_TARGET = pd.Timestamp("2026-11-12 22:00", tz="UTC")     # 00:00 Helsinki


class FakeModel:
    def predict(self, X):
        p50 = X["wind"] * -1.0 + 80
        return pd.DataFrame({"p10": p50 - 10, "p50": p50, "p90": p50 + 10}, index=X.index)

    def contributions(self, X):
        cols = {f"f{i}": float(i) for i in range(12)}
        cols["wind"] = -40.0
        cols["bias"] = 60.0
        return pd.DataFrame(cols, index=X.index)

    @classmethod
    def load(cls, path):
        return cls()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("FORECAST_ENABLED", "false")
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    def override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    api = FastAPI()
    api.include_router(forecast.router)
    api.dependency_overrides[get_db] = override
    training.clear_model_cache()
    db = Session()
    yield TestClient(api), db, monkeypatch
    db.close()
    training.clear_model_cache()


def seed(db, version="m1", activate=True, hours=168, corrected_shift=5.0, issued=ISSUED):
    store.register_model(db, version, "unused.model", training.TRAIN_START, issued,
                         holdout_mae=8.0, metrics={"reason": "test"}, activate=activate)
    idx = pd.date_range(FIRST_TARGET, periods=hours, freq="h")
    df = pd.DataFrame({"p10": 90.0, "p50": 100.0, "p90": 120.0}, index=idx)
    df["p50_corrected"] = 100.0 + corrected_shift
    store.save_predictions(db, df, issued, version)
    # the first day-ahead day is already known
    store.upsert_spot_prices(db, pd.DataFrame({
        "timestamp": idx[:24], "zone": "FI", "price_eur_mwh": 50.0,
    }))


def test_forecast_404_when_nothing_issued(setup):
    client, _, _ = setup
    r = client.get("/forecast")
    assert r.status_code == 404
    assert "No forecast" in r.json()["detail"]


def test_forecast_default_unit_is_c_kwh_with_vat(setup):
    client, db, _ = setup
    seed(db)
    body = client.get("/forecast").json()
    assert body["unit"] == "c_kwh_vat" and body["model_version"] == "m1"
    first = body["points"][0]
    # corrected p50 105 EUR/MWh -> 10.5 c/kWh * 1.255; band shifted by +5
    assert first["p50"] == pytest.approx(round(105 / 10 * 1.255, 3))
    assert first["p10"] == pytest.approx(round(95 / 10 * 1.255, 3))
    assert first["p90"] == pytest.approx(round(125 / 10 * 1.255, 3))
    assert first["known_price"] == pytest.approx(round(50 / 10 * 1.255, 3))
    assert body["points"][30]["known_price"] is None
    assert first["p10"] <= first["p50"] <= first["p90"]


def test_forecast_eur_mwh_and_days(setup):
    client, db, _ = setup
    seed(db)
    body = client.get("/forecast", params={"unit": "eur_mwh", "days": 2}).json()
    assert body["points"][0]["p50"] == 105.0 and body["points"][0]["p90"] == 125.0
    # targets before issued_at + 48 h
    assert len(body["points"]) == 39  # 22:00 day 0 .. 12:00 day 2 UTC
    assert client.get("/forecast", params={"unit": "bogus"}).status_code == 422
    assert client.get("/forecast", params={"days": 8}).status_code == 422


def test_forecast_uncorrected_falls_back_to_raw_p50(setup):
    client, db, _ = setup
    store.register_model(db, "m1", "x", training.TRAIN_START, ISSUED, holdout_mae=1.0, activate=True)
    idx = pd.date_range(FIRST_TARGET, periods=3, freq="h")
    store.save_predictions(db, pd.DataFrame({"p10": 1.0, "p50": 2.0, "p90": 3.0}, index=idx), ISSUED, "m1")
    pts = client.get("/forecast", params={"unit": "eur_mwh"}).json()["points"]
    assert [p["p50"] for p in pts] == [2.0, 2.0, 2.0]


def test_forecast_timezone_conversion(setup):
    client, db, _ = setup
    seed(db)
    body = client.get("/forecast", params={"timezone_str": "Europe/Helsinki"}).json()
    assert body["points"][0]["timestamp"] == "2026-11-13T00:00:00+02:00"
    assert body["issued_at"] == "2026-11-12T14:45:00+02:00"
    utc = client.get("/forecast").json()
    assert utc["points"][0]["timestamp"] in ("2026-11-12T22:00:00Z", "2026-11-12T22:00:00+00:00")
    assert client.get("/forecast", params={"timezone_str": "Mars/Base"}).status_code == 400


def test_forecast_prefers_latest_issue_of_active_model(setup):
    client, db, _ = setup
    seed(db, version="old")
    later = datetime(2026, 11, 12, 13, 45, tzinfo=timezone.utc)
    seed(db, version="new", activate=False, issued=later)  # newer, but not active
    assert client.get("/forecast").json()["model_version"] == "old"
    store.activate_model(db, "new")
    assert client.get("/forecast").json()["model_version"] == "new"


def test_forecast_falls_back_to_other_model_after_promotion(setup):
    client, db, _ = setup
    seed(db, version="old")
    store.register_model(db, "fresh", "x", training.TRAIN_START, ISSUED, holdout_mae=1.0, activate=True)
    assert client.get("/forecast").json()["model_version"] == "old"


def test_backtest_predictions_are_not_served(setup):
    client, db, _ = setup
    store.register_model(db, "bt", "x", training.TRAIN_START, ISSUED, holdout_mae=1.0, activate=True)
    idx = pd.date_range(FIRST_TARGET, periods=3, freq="h")
    store.save_predictions(db, pd.DataFrame({"p10": 1.0, "p50": 2.0, "p90": 3.0}, index=idx),
                           ISSUED, "bt", is_backtest=True)
    assert client.get("/forecast").status_code == 404


def test_accuracy_endpoint(setup):
    client, db, _ = setup
    seed(db)
    store.fill_actuals(db)
    body = client.get("/forecast/accuracy").json()
    assert set(body["windows"]) == {"14d", "28d"}
    assert len(body["windows"]["14d"]) == 7
    assert body["holdout_mae"] == 8.0 and body["model_version"] == "m1"
    assert body["drift"]["drift"] is False
    assert body["unit"] == "eur_mwh"


def test_drivers_endpoint(setup):
    client, db, monkeypatch = setup
    seed(db)
    monkeypatch.setattr(training, "PriceModel", FakeModel)

    def fake_frame(db, issued_at, horizon_days=7):
        idx = pd.date_range(FIRST_TARGET, periods=horizon_days * 24, freq="h")
        X = pd.DataFrame({f"f{i}": 0.0 for i in range(12)}, index=idx)
        X["wind"] = 30.0
        return X

    monkeypatch.setattr(forecast.features, "build_prediction_frame", fake_frame)
    r = client.get("/forecast/drivers", params={"timestamp": "2026-11-13T05:30:00",
                                                "timezone_str": "Europe/Helsinki"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["timestamp"] == "2026-11-13T05:00:00+02:00"
    assert body["base_value"] == 60.0 and body["p50"] == 50.0
    assert body["unit"] == "eur_mwh"
    assert body["contribution_unit"] == "asinh_eur_mwh_div_10"
    feats = [c["feature"] for c in body["contributions"]]
    assert len(feats) == 10 and feats[0] == "wind" and "bias" not in feats
    assert body["contributions"][0]["value"] == 30.0
    out = client.get("/forecast/drivers", params={"timestamp": "2027-01-01T00:00:00"})
    assert out.status_code == 404


def test_drivers_404_without_model(setup):
    client, _, _ = setup
    assert client.get("/forecast/drivers", params={"timestamp": "2026-11-13T00:00:00"}).status_code == 404


def test_status_endpoint(setup):
    client, db, _ = setup
    seed(db)
    body = client.get("/forecast/status").json()
    assert body["scheduler"]["enabled"] is False and body["scheduler"]["running"] is False
    assert body["active_model"]["version"] == "m1"
    assert body["active_model"]["metrics"] == {"reason": "test"}
    fi = body["coverage"]["spot_prices_fi"]
    assert fi["rows"] == 24 and fi["min"].startswith("2026-11-12T22:00:00")
    assert body["coverage"]["predictions_live"]["rows"] == 168
    assert body["latest_issued_at"].startswith("2026-11-12T12:45:00")


def test_scheduler_disabled_by_env(monkeypatch):
    monkeypatch.setenv("FORECAST_ENABLED", "false")
    assert scheduler.start_scheduler() is None
    assert scheduler.get_status()["running"] is False


def test_scheduler_job_wrapper_records_errors(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 's.db'}")
    monkeypatch.setattr(scheduler.database, "SessionLocal", sessionmaker(bind=engine))

    def boom(db):
        raise ValueError("nope")

    scheduler._run("unit_boom", boom)
    scheduler._run("unit_ok", lambda db: {"x": 1})
    runs = scheduler.get_status()["last_runs"]
    assert runs["unit_boom"]["status"] == "error" and "nope" in runs["unit_boom"]["error"]
    assert runs["unit_ok"]["status"] == "ok" and runs["unit_ok"]["result"] == {"x": 1}
    json.dumps(runs)


def test_scheduler_registers_jobs(monkeypatch):
    monkeypatch.setenv("FORECAST_ENABLED", "true")
    monkeypatch.setattr(scheduler, "_startup", lambda db: None)
    try:
        sched = scheduler.start_scheduler()
        ids = {j["id"] for j in scheduler.get_status()["jobs"]}
        assert {"snapshot", "daily_refresh", "daily_predict", "weekly_retrain", "startup"} <= ids
        assert all(j.max_instances == 1 and j.coalesce for j in sched.get_jobs())
        assert str(sched.timezone) == "Europe/Helsinki"
    finally:
        scheduler.stop_scheduler()
