from datetime import datetime, timezone

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import app.models  # noqa: F401
import app.forecasting.models  # noqa: F401
from app.database import Base
from app.forecasting import ingest, store
from app.forecasting import series as S
from app.forecasting.models import DriverData, ForecastSnapshot, SpotPriceRaw


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _hours(start, end):
    return pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="h", inclusive="left")


def _driver_frame(start, end, names, value=1.0):
    idx = _hours(start, end)
    return pd.DataFrame([
        {"timestamp": ts, "series": name, "value": value} for ts in idx for name in names
    ])


class FakeEntsoe:
    def __init__(self, available=True, fail_prices=False):
        self.available = available
        self.fail_prices = fail_prices
        self.price_calls = []
        self.driver_calls = []

    def is_available(self):
        return self.available

    def fetch_prices(self, start, end, zones=S.ZONES):
        self.price_calls.append((start, end))
        if self.fail_prices:
            raise RuntimeError("entsoe down")
        idx = _hours(start, end)
        return pd.DataFrame([
            {"timestamp": ts, "zone": z, "price_eur_mwh": 10.0, "resolution_minutes": 60}
            for ts in idx for z in zones
        ])

    def fetch_drivers(self, start, end):
        self.driver_calls.append((start, end))
        return _driver_frame(start, end, [S.LOAD_FORECAST_MW])

    def fetch_forecasts(self, start, end):
        return _driver_frame(start, end, [S.LOAD_FORECAST_MW], value=5.0)


class FakeFingrid:
    def __init__(self, available=True):
        self.available = available
        self.calls = []

    def is_available(self):
        return self.available

    def fetch_drivers(self, start, end):
        self.calls.append((start, end))
        return _driver_frame(start, end, [S.CONSUMPTION_ACTUAL_MW, S.NUCLEAR_ACTUAL_MW, S.WIND_FORECAST_MW])

    def fetch_forecasts(self, start, end):
        return _driver_frame(start, end, [S.WIND_FORECAST_MW], value=7.0)


class FakeOpenMeteo:
    def __init__(self):
        self.hist_calls = []
        self.fc_calls = []

    def fetch_historical_forecasts(self, start, end):
        self.hist_calls.append((start, end))
        return _driver_frame(start, end, [S.TEMPERATURE_FORECAST_C], value=-3.0)

    def fetch_forecasts(self, start, end):
        self.fc_calls.append((start, end))
        return _driver_frame(start, end, [S.TEMPERATURE_FORECAST_C], value=-4.0)


def _count(db, model):
    return db.execute(select(func.count()).select_from(model)).scalar()


def test_month_chunks():
    chunks = ingest._month_chunks(pd.Timestamp("2023-01-15", tz="UTC"), pd.Timestamp("2023-03-10", tz="UTC"))
    assert [(a.isoformat(), b.isoformat()) for a, b in chunks] == [
        ("2023-01-15T00:00:00+00:00", "2023-02-01T00:00:00+00:00"),
        ("2023-02-01T00:00:00+00:00", "2023-03-01T00:00:00+00:00"),
        ("2023-03-01T00:00:00+00:00", "2023-03-10T00:00:00+00:00"),
    ]


def test_backfill_chunks_and_stores(db):
    entsoe, fingrid, om = FakeEntsoe(), FakeFingrid(), FakeOpenMeteo()
    sources = {"entsoe": entsoe, "fingrid": fingrid, "openmeteo": om}
    result = ingest.backfill(db, utc(2023, 1, 30), utc(2023, 2, 2), sources=sources)

    assert len(entsoe.price_calls) == 2  # Jan 30-Feb 1, Feb 1-2
    assert len(om.hist_calls) == 2
    assert fingrid.calls == [(utc(2023, 1, 30), utc(2023, 2, 2))]  # missing months merged
    hours = 72
    assert result["counts"]["entsoe_prices"] == hours * len(S.ZONES)
    assert result["counts"]["fingrid"] == hours * 3  # consumption, nuclear, wind
    assert result["failures"] == []
    assert _count(db, SpotPriceRaw) == hours * len(S.ZONES)
    drivers = store.load_drivers(db, utc(2023, 1, 30), utc(2023, 2, 2))
    assert set(drivers.columns) == {S.LOAD_FORECAST_MW, S.CONSUMPTION_ACTUAL_MW, S.NUCLEAR_ACTUAL_MW,
                                    S.WIND_FORECAST_MW, S.TEMPERATURE_FORECAST_C}
    assert len(drivers) == hours


def test_backfill_resumes_and_skips_stored_chunks(db):
    sources = {"entsoe": FakeEntsoe(), "fingrid": FakeFingrid(), "openmeteo": FakeOpenMeteo()}
    ingest.backfill(db, utc(2023, 1, 1), utc(2023, 2, 1), sources=sources)

    entsoe, fingrid, om = FakeEntsoe(), FakeFingrid(), FakeOpenMeteo()
    result = ingest.backfill(db, utc(2023, 1, 1), utc(2023, 3, 1),
                             sources={"entsoe": entsoe, "fingrid": fingrid, "openmeteo": om})
    # January already stored -> only February fetched
    assert [c[0] for c in entsoe.price_calls] == [utc(2023, 2, 1)]
    assert [c[0] for c in fingrid.calls] == [utc(2023, 2, 1)]
    assert result["skipped_chunks"] == {"entsoe_prices": 1, "entsoe_drivers": 1,
                                        "fingrid": 1, "openmeteo": 1}


def test_backfill_refetches_partially_covered_chunk(db):
    # Only 50% of January stored for openmeteo -> refetch
    store.upsert_driver_data(db, _driver_frame(utc(2023, 1, 1), utc(2023, 1, 16), [S.TEMPERATURE_FORECAST_C]),
                             source="openmeteo")
    om = FakeOpenMeteo()
    ingest.backfill(db, utc(2023, 1, 1), utc(2023, 2, 1), sources={"openmeteo": om})
    assert len(om.hist_calls) == 1


def test_unavailable_source_skipped(db):
    fingrid = FakeFingrid(available=False)
    result = ingest.backfill(db, utc(2023, 1, 1), utc(2023, 1, 2),
                             sources={"entsoe": FakeEntsoe(), "fingrid": fingrid, "openmeteo": FakeOpenMeteo()})
    assert fingrid.calls == []
    assert "fingrid" in result["unavailable"]
    assert result["counts"]["fingrid"] == 0
    assert result["counts"]["openmeteo"] == 24


def test_failing_source_does_not_abort(db):
    entsoe = FakeEntsoe(fail_prices=True)
    result = ingest.backfill(db, utc(2023, 1, 31), utc(2023, 2, 2),
                             sources={"entsoe": entsoe, "fingrid": FakeFingrid(), "openmeteo": FakeOpenMeteo()})
    assert len(result["failures"]) == 2
    assert all(f["job"] == "entsoe_prices" for f in result["failures"])
    assert "entsoe down" in result["failures"][0]["error"]
    assert len(entsoe.price_calls) == 2  # retried for the next chunk
    assert result["counts"]["entsoe_drivers"] == 48
    assert result["counts"]["fingrid"] == 48 * 3
    assert _count(db, SpotPriceRaw) == 0


def test_refresh_recent_splits_openmeteo(db):
    entsoe, om = FakeEntsoe(), FakeOpenMeteo()
    now = utc(2024, 5, 10, 12, 30)
    result = ingest.refresh_recent(db, days_back=1, days_ahead=1, now=now,
                                   sources={"entsoe": entsoe, "openmeteo": om})
    assert om.hist_calls == [(utc(2024, 5, 9, 12), utc(2024, 5, 10, 12))]
    assert om.fc_calls == [(utc(2024, 5, 10, 12), utc(2024, 5, 11, 12))]
    assert entsoe.price_calls == [(utc(2024, 5, 9, 12), utc(2024, 5, 11, 12))]
    assert result["counts"]["openmeteo"] == 48
    assert "fingrid" in result["unavailable"]
    # idempotent
    ingest.refresh_recent(db, days_back=1, days_ahead=1, now=now, sources={"entsoe": entsoe, "openmeteo": om})
    assert _count(db, SpotPriceRaw) == 48 * len(S.ZONES)


def test_capture_snapshot_stores_snapshot_and_drivers(db):
    class Broken:
        def is_available(self):
            return True

        def fetch_forecasts(self, start, end):
            raise RuntimeError("boom")

    sources = {"entsoe": FakeEntsoe(), "fingrid": FakeFingrid(), "openmeteo": FakeOpenMeteo(),
               "broken": Broken()}
    issued = utc(2024, 5, 10, 12, 45)
    n = ingest.capture_snapshot(db, issued_at=issued, sources=sources)

    hours = 8 * 24
    assert n == hours * 3
    assert _count(db, ForecastSnapshot) == hours * 3
    issued_vals = {r for (r,) in db.execute(select(ForecastSnapshot.issued_at).distinct())}
    assert issued_vals == {utc(2024, 5, 10, 12)}

    snap = store.load_snapshot_asof(db, utc(2024, 5, 10, 13), utc(2024, 5, 10, 12), utc(2024, 5, 18, 12))
    assert snap[S.WIND_FORECAST_MW].eq(7.0).all()

    drivers = store.load_drivers(db, utc(2024, 5, 10, 12), utc(2024, 5, 18, 12))
    assert drivers[S.TEMPERATURE_FORECAST_C].eq(-4.0).all()
    src = db.execute(select(DriverData.source).where(DriverData.series == S.WIND_FORECAST_MW)).scalars().first()
    assert src == "fingrid"


def test_capture_snapshot_updates_latest_driver_values(db):
    store.upsert_driver_data(db, _driver_frame(utc(2024, 5, 10, 12), utc(2024, 5, 10, 14),
                                               [S.TEMPERATURE_FORECAST_C], value=99.0), source="openmeteo")
    ingest.capture_snapshot(db, issued_at=utc(2024, 5, 10, 12), sources={"openmeteo": FakeOpenMeteo()})
    drivers = store.load_drivers(db, utc(2024, 5, 10, 12), utc(2024, 5, 10, 14))
    assert drivers[S.TEMPERATURE_FORECAST_C].eq(-4.0).all()


def test_get_sources_skips_unavailable_and_broken(monkeypatch):
    def broken():
        raise NotImplementedError

    monkeypatch.setattr(ingest, "_BUILDERS", {
        "entsoe": lambda: FakeEntsoe(),
        "fingrid": lambda: FakeFingrid(available=False),
        "openmeteo": broken,
    })
    assert list(ingest.get_sources()) == ["entsoe"]


def test_backfill_merges_missing_months_into_large_windows(db):
    sources = {"entsoe": FakeEntsoe(), "fingrid": FakeFingrid(), "openmeteo": FakeOpenMeteo()}
    ingest.backfill(db, utc(2023, 3, 1), utc(2023, 4, 1), sources=sources)  # March stored

    fingrid = FakeFingrid()
    result = ingest.backfill(db, utc(2023, 1, 1), utc(2023, 10, 1),
                             sources={"entsoe": FakeEntsoe(), "fingrid": fingrid, "openmeteo": FakeOpenMeteo()})
    # Jan-Feb, then Apr-Sep capped at 6 months; March is skipped
    assert fingrid.calls == [
        (utc(2023, 1, 1), utc(2023, 3, 1)),
        (utc(2023, 4, 1), utc(2023, 10, 1)),
    ]
    assert result["skipped_chunks"]["fingrid"] == 1
    assert result["failures"] == []
