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


# ---------------------------------------------------------------- previous runs

ALL_SERIES = {S.TEMPERATURE_FORECAST_C, S.WIND_SPEED_100M_FORECAST_MS, S.SOLAR_RADIATION_FORECAST_WM2}


def _raw(base_var, loc, hour):
    """Deterministic, location-asymmetric raw value (same for every API / lead)."""
    scale = {"temperature_2m": 1.0, "shortwave_radiation": 37.0, "wind_speed_100m": 0.7}[base_var]
    return scale * ((loc * 7 + hour * 3) % 11) + 0.1 * loc


class PrevRunsSession:
    """Serves both the historical-forecast and previous-runs APIs from the same raw table.

    previous_dayN values = raw + lead_offset * N. ``fail_start_dates`` -> HTTP 400 for
    chunks starting on those dates; ``missing_leads`` -> variable absent from response;
    ``null_leads`` -> variable present but all null.
    """

    def __init__(self, n_hours=4, lead_offset=0.0, fail_start_dates=(), missing_leads=(),
                 null_leads=()):
        self.calls = []
        self.n = n_hours
        self.lead_offset = lead_offset
        self.fail_start_dates = set(fail_start_dates)
        self.missing_leads = set(missing_leads)
        self.null_leads = set(null_leads)

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params)))
        if params.get("start_date") in self.fail_start_dates:
            return FakeResponse(400, {"error": True, "reason": "boom"})
        times = _times(params.get("start_date", "2024-01-01") + "T00:00", self.n)
        locs = []
        for i, _ in enumerate(params["latitude"].split(",")):
            vals = {}
            for v in params["hourly"].split(","):
                if "_previous_day" in v:
                    base, lead = v.split("_previous_day")
                    lead = int(lead)
                    if lead in self.missing_leads:
                        continue
                    if lead in self.null_leads:
                        vals[v] = [None] * self.n
                        continue
                    vals[v] = [_raw(base, i, h) + self.lead_offset * lead for h in range(self.n)]
                else:
                    vals[v] = [_raw(v, i, h) for h in range(self.n)]
            locs.append(_location(times, vals))
        return FakeResponse(200, locs if len(locs) > 1 else locs[0])


P_START = datetime(2024, 6, 1, tzinfo=timezone.utc)
P_END = datetime(2024, 6, 1, 4, tzinfo=timezone.utc)


def test_previous_runs_long_format_all_leads():
    session = PrevRunsSession(lead_offset=100.0)
    df = OpenMeteoClient(session=session).fetch_previous_runs(P_START, P_END)
    assert list(df.columns) == ["timestamp", "series", "lead_days", "value"]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert df["lead_days"].dtype.kind == "i"
    assert sorted(df["lead_days"].unique()) == [1, 2, 3, 4, 5, 6, 7]
    assert set(df["series"]) == ALL_SERIES
    assert len(df) == 7 * 3 * 4  # leads x series x hours
    assert not df.duplicated(["timestamp", "series", "lead_days"]).any()

    # requests: one per point group, all previous_day1..7 variables of that group
    assert len(session.calls) == 2
    for url, params in session.calls:
        assert url == OpenMeteoClient.PREVIOUS_RUNS_URL
        assert params["timezone"] == "UTC" and params["wind_speed_unit"] == "ms"
        assert params["start_date"] == params["end_date"] == "2024-06-01"
        assert "models" not in params  # best_match default
    hourly = sorted(v for _, p in session.calls for v in p["hourly"].split(","))
    expected = sorted(f"{v}_previous_day{n}" for n in range(1, 8)
                      for v in ("temperature_2m", "shortwave_radiation", "wind_speed_100m"))
    assert hourly == expected
    city_lats = [p["latitude"] for _, p in session.calls if "temperature_2m_previous_day1" in p["hourly"]]
    assert len(city_lats[0].split(",")) == len(om.CITIES)

    # values distinguish leads: lead N = lead-0 aggregate + 100*N (aggregates are means)
    wide = df.pivot_table(index=["timestamp", "series"], columns="lead_days", values="value")
    for n in range(2, 8):
        assert ((wide[n] - wide[1]) - 100.0 * (n - 1)).abs().max() == pytest.approx(0.0, abs=1e-9)


def test_previous_runs_aggregation_identical_to_historical():
    """Same raw numbers -> identical aggregates (weights, means, missing handling)."""
    session = PrevRunsSession(lead_offset=0.0)
    client = OpenMeteoClient(session=session)
    hist = client.fetch_historical_forecasts(P_START, P_END)
    prev = client.fetch_previous_runs(P_START, P_END, lead_days=(1, 4, 7))
    hist_wide = hist.pivot(index="timestamp", columns="series", values="value").sort_index()
    for n in (1, 4, 7):
        lead = prev[prev["lead_days"] == n]
        lead_wide = lead.pivot(index="timestamp", columns="series", values="value").sort_index()
        pd.testing.assert_frame_equal(lead_wide, hist_wide)
    # and the temperature aggregate really is population-weighted (not a plain mean)
    weights = [c[3] for c in om.CITIES]
    t0 = hist_wide.index[0]
    raw = [_raw("temperature_2m", i, 0) for i in range(len(om.CITIES))]
    assert hist_wide.loc[t0, S.TEMPERATURE_FORECAST_C] == pytest.approx(
        sum(r * w for r, w in zip(raw, weights)) / sum(weights))
    assert hist_wide.loc[t0, S.TEMPERATURE_FORECAST_C] != pytest.approx(sum(raw) / len(raw))


def test_previous_runs_chunks_and_skips_failing_chunk():
    session = PrevRunsSession(fail_start_dates={"2024-04-01"})
    df = OpenMeteoClient(session=session).fetch_previous_runs(
        datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2024, 10, 1, tzinfo=timezone.utc),
        lead_days=(1, 2))
    windows = sorted({(p["start_date"], p["end_date"]) for _, p in session.calls})
    assert windows == [("2024-01-01", "2024-03-31"), ("2024-04-01", "2024-06-30"),
                       ("2024-07-01", "2024-09-30")]
    days = sorted(df["timestamp"].dt.strftime("%Y-%m-%d").unique())
    assert days == ["2024-01-01", "2024-07-01"]  # fake serves n_hours from each chunk start
    assert sorted(df["lead_days"].unique()) == [1, 2]


def test_previous_runs_clips_to_data_start_and_now():
    session = PrevRunsSession()
    OpenMeteoClient(session=session).fetch_previous_runs(
        datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2024, 2, 1, tzinfo=timezone.utc))
    assert min(p["start_date"] for _, p in session.calls) == "2024-01-01"

    session = PrevRunsSession()
    future = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)
    df = OpenMeteoClient(session=session).fetch_previous_runs(future, future + pd.Timedelta(days=1))
    assert df.empty and session.calls == []
    assert list(df.columns) == ["timestamp", "series", "lead_days", "value"]


def test_previous_runs_skips_unavailable_leads():
    session = PrevRunsSession(missing_leads={6}, null_leads={7})
    df = OpenMeteoClient(session=session).fetch_previous_runs(
        P_START, P_END, lead_days=(1, 6, 7, 9))
    assert sorted(df["lead_days"].unique()) == [1]
    # lead 9 is never requested
    assert not any("previous_day9" in p["hourly"] for _, p in session.calls)


def test_previous_runs_all_chunks_fail_returns_empty():
    df = OpenMeteoClient(session=PrevRunsSession(fail_start_dates={"2024-06-01"})).fetch_previous_runs(
        P_START, P_END)
    assert df.empty
    assert list(df.columns) == ["timestamp", "series", "lead_days", "value"]
