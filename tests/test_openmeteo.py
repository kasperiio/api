from datetime import datetime, timezone

import pandas as pd
import pytest

from app.forecasting import series as S
from app.forecasting.sources import openmeteo as om
from app.forecasting.sources.openmeteo import OpenMeteoClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = ""

    def json(self):
        return self._payload


def _location(times, values):
    """One Open-Meteo location object; values: {var: list}."""
    hourly = {"time": times}
    hourly.update(values)
    return {"latitude": 60.0, "longitude": 24.0, "utc_offset_seconds": 0,
            "timezone": "GMT", "hourly": hourly}


def _times(start, n):
    t0 = pd.Timestamp(start)
    return [(t0 + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(n)]


class FakeSession:
    def __init__(self, n_hours=3, start="2024-01-01T00:00", fail=False):
        self.calls = []
        self.n = n_hours
        self.start = start
        self.fail = fail

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        if self.fail:
            return FakeResponse(400, {"error": True, "reason": "bad"})
        lats = params["latitude"].split(",")
        variables = params["hourly"].split(",")
        times = _times(self.start, self.n)
        locs = []
        for i, _ in enumerate(lats):
            vals = {}
            for v in variables:
                if v == "temperature_2m":
                    vals[v] = [float(i)] * self.n  # city i has temperature i
                elif v == "shortwave_radiation":
                    vals[v] = [10.0 * i] * self.n
                elif v == "wind_speed_100m":
                    vals[v] = [float(i + 1)] * self.n
            locs.append(_location(times, vals))
        return FakeResponse(200, locs if len(locs) > 1 else locs[0])


START = datetime(2024, 1, 1, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, 3, tzinfo=timezone.utc)


def test_historical_shape_and_aggregates():
    session = FakeSession()
    df = OpenMeteoClient(session=session).fetch_historical_forecasts(START, END)
    assert list(df.columns) == ["timestamp", "series", "value"]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert set(df["series"]) == {
        S.TEMPERATURE_FORECAST_C, S.WIND_SPEED_100M_FORECAST_MS, S.SOLAR_RADIATION_FORECAST_WM2
    }
    wide = df.pivot(index="timestamp", columns="series", values="value")
    assert len(wide) == 3
    assert (wide.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()

    weights = [c[3] for c in om.CITIES]
    expected_temp = sum(i * w for i, w in enumerate(weights)) / sum(weights)
    assert wide[S.TEMPERATURE_FORECAST_C].iloc[0] == pytest.approx(expected_temp)
    n_cities = len(om.CITIES)
    assert wide[S.SOLAR_RADIATION_FORECAST_WM2].iloc[0] == pytest.approx(
        10.0 * sum(range(n_cities)) / n_cities)
    n_wind = len(om.WIND_POINTS)
    assert wide[S.WIND_SPEED_100M_FORECAST_MS].iloc[0] == pytest.approx(
        sum(range(1, n_wind + 1)) / n_wind)


def test_request_params():
    session = FakeSession()
    OpenMeteoClient(session=session).fetch_historical_forecasts(START, END)
    assert len(session.calls) == 2  # cities + wind points, one request each
    for url, params in session.calls:
        assert url == OpenMeteoClient.HISTORICAL_URL
        assert params["timezone"] == "UTC"
        assert params["wind_speed_unit"] == "ms"
        assert params["start_date"] == "2024-01-01"
        assert params["end_date"] == "2024-01-01"
        assert len(params["latitude"].split(",")) == len(params["longitude"].split(","))


def test_historical_chunks_long_range():
    session = FakeSession()
    OpenMeteoClient(session=session).fetch_historical_forecasts(
        datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2023, 7, 1, tzinfo=timezone.utc))
    windows = sorted({(p["start_date"], p["end_date"]) for _, p in session.calls})
    assert windows == [("2023-01-01", "2023-03-31"), ("2023-04-01", "2023-06-30")]


def test_weighted_mean_ignores_missing_values():
    idx = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    a = pd.Series([1.0, float("nan")], index=idx)
    b = pd.Series([3.0, 5.0], index=idx)
    out = OpenMeteoClient._weighted_mean([a, b], pd.Series([1.0, 3.0]))
    assert out.tolist() == [pytest.approx(2.5), pytest.approx(5.0)]


def test_forecast_uses_forecast_api_and_filters_range():
    session = FakeSession(n_hours=48, start="2024-01-01T00:00")
    df = OpenMeteoClient(session=session).fetch_forecasts(
        datetime(2024, 1, 1, 6, tzinfo=timezone.utc), datetime(2024, 1, 1, 12, tzinfo=timezone.utc))
    url, params = session.calls[0]
    assert url == OpenMeteoClient.FORECAST_URL
    assert params["forecast_days"] == 16
    assert 0 <= params["past_days"] <= 92
    assert df["timestamp"].min() == pd.Timestamp("2024-01-01T06:00", tz="UTC")
    assert df["timestamp"].max() == pd.Timestamp("2024-01-01T11:00", tz="UTC")
    assert len(df) == 6 * 3


def test_http_error_returns_empty_frame():
    df = OpenMeteoClient(session=FakeSession(fail=True)).fetch_forecasts(START, END)
    assert df.empty
    assert list(df.columns) == ["timestamp", "series", "value"]
    df = OpenMeteoClient(session=FakeSession(fail=True)).fetch_historical_forecasts(START, END)
    assert df.empty
