import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.models  # noqa: F401
import app.forecasting.models  # noqa: F401
from app.database import Base
from app.forecasting import store

HOURS = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
END = HOURS[-1] + pd.Timedelta("1h")


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _prices(values):
    return pd.DataFrame({
        "timestamp": HOURS, "zone": "FI",
        "price_eur_mwh": values, "resolution_minutes": 60,
    })


def test_spot_prices_upsert_and_load(db):
    store.upsert_spot_prices(db, _prices([1, 2, 3, 4]))
    store.upsert_spot_prices(db, _prices([5, 6, 7, 8]))
    loaded = store.load_spot_prices(db, HOURS[0], END)
    assert list(loaded["FI"]) == [5, 6, 7, 8]
    assert str(loaded.index.tz) == "UTC"


def test_outage_moved_outside_query_window_supersedes_old_revision(db):
    from test_outages import event, ts
    from app.forecasting.outages import outage_features
    import numpy as np

    store.upsert_outages(db, pd.DataFrame([
        event("moved", 1, -10, 5, 10, 1600),
        event("moved", 2, -5, 100, 110, 1600),
    ]))
    events = store.load_outages(db, ts(0), ts(24), published_before=ts(0))
    assert set(events.revision) == {1, 2}
    targets = pd.date_range(ts(1), ts(24), freq="h")
    values = outage_features(events, pd.DatetimeIndex([ts(0)]), targets,
                             np.zeros(len(targets), dtype=int))
    assert values["nuclear_unavailable_mw"].max() == 0


def test_spot_prices_aggregate_quarters_to_hourly_mean(db):
    quarters = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    store.upsert_spot_prices(db, pd.DataFrame({
        "timestamp": quarters, "zone": "FI",
        "price_eur_mwh": [10, 20, 30, 40], "resolution_minutes": 15,
    }))
    loaded = store.load_spot_prices(db, quarters[0], END)
    assert list(loaded["FI"]) == [25]


def test_naive_timestamps_rejected(db):
    df = _prices([1, 2, 3, 4])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    with pytest.raises(ValueError):
        store.upsert_spot_prices(db, df)


def test_snapshot_asof_uses_latest_snapshot_before_issue(db):
    frame = pd.DataFrame({"timestamp": HOURS, "series": "wind_forecast_mw", "value": 1.0})
    store.insert_snapshot(db, frame, HOURS[0])
    store.insert_snapshot(db, frame.assign(value=2.0), HOURS[2])
    early = store.load_snapshot_asof(db, HOURS[1], HOURS[0], END)
    late = store.load_snapshot_asof(db, HOURS[3], HOURS[0], END)
    assert set(early["wind_forecast_mw"]) == {1.0}
    assert set(late["wind_forecast_mw"]) == {2.0}


def test_predictions_get_actuals_filled(db):
    store.upsert_spot_prices(db, _prices([1, 2, 3, 4]))
    preds = pd.DataFrame({"p10": 0.0, "p50": 1.0, "p90": 2.0}, index=HOURS)
    issued = HOURS[0] - pd.Timedelta("1h")
    store.save_predictions(db, preds, issued, "v1")
    store.save_predictions(db, preds, issued, "v1")  # re-issue replaces, no duplicates
    assert store.fill_actuals(db) == 4
    scored = store.load_predictions(db, HOURS[0], END, scored_only=True)
    assert len(scored) == 4
    assert list(scored.sort_values("target_ts")["actual"]) == [1, 2, 3, 4]
    assert list(scored.sort_values("target_ts")["horizon_hours"]) == [1, 2, 3, 4]


def test_only_one_active_model(db):
    store.register_model(db, "v1", "a", HOURS[0], END, 10.0, activate=True)
    store.register_model(db, "v2", "b", HOURS[0], END, 8.0, activate=True)
    assert store.get_active_model(db).version == "v2"


def test_reissue_keeps_actual_and_updates_values(db):
    store.upsert_spot_prices(db, _prices([1, 2, 3, 4]))
    preds = pd.DataFrame({"p10": 0.0, "p50": 1.0, "p90": 2.0}, index=HOURS)
    issued = HOURS[0] - pd.Timedelta("1h")
    store.save_predictions(db, preds, issued, "v1")
    store.fill_actuals(db)
    store.save_predictions(db, preds.assign(p50=1.5, p50_corrected=1.7), issued, "v1")
    scored = store.load_predictions(db, HOURS[0], END, scored_only=True)
    assert len(scored) == 4
    assert set(scored["p50"]) == {1.5}
    assert set(scored["p50_corrected"]) == {1.7}


def test_driver_upsert_overwrites_value(db):
    frame = pd.DataFrame({"timestamp": HOURS, "series": "wind_forecast_mw", "value": 1.0})
    store.upsert_driver_data(db, frame, "a")
    store.upsert_driver_data(db, frame.assign(value=3.0), "b")
    assert set(store.load_drivers(db, HOURS[0], END)["wind_forecast_mw"]) == {3.0}


def test_outages_roundtrip_and_published_filter(db):
    frame = pd.DataFrame([
        dict(mrid="m1", revision=1, published_at=HOURS[0], kind="generation", business_type="planned",
             status="active", resource="OL3", border="", psr_type="B14", nominal_mw=1600.0,
             start=HOURS[1], end=HOURS[3], available_mw=0.0, unavailable_mw=1600.0),
        dict(mrid="m1", revision=2, published_at=HOURS[2], kind="generation", business_type="planned",
             status="cancelled", resource="OL3", border="", psr_type="B14", nominal_mw=1600.0,
             start=HOURS[1], end=HOURS[3], available_mw=0.0, unavailable_mw=1600.0),
    ])
    assert store.upsert_outages(db, frame) == 2
    assert len(store.load_outages(db, HOURS[0], END)) == 2
    known = store.load_outages(db, HOURS[0], END, published_before=HOURS[1])
    assert list(known["revision"]) == [1]
    assert str(known["start"].dt.tz) == "UTC"
