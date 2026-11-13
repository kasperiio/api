from datetime import datetime, timezone

import pandas as pd
import pytest
import requests

from app.forecasting import series as S
from app.forecasting.sources import fingrid as fg
from app.forecasting.sources.fingrid import FingridClient


class FakeResponse:
    def __init__(self, status_code=200, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _points(dataset_id, start, minutes, count, value_fn):
    t0 = pd.Timestamp(start)
    out = []
    for i in range(count):
        s = t0 + pd.Timedelta(minutes=minutes * i)
        e = s + pd.Timedelta(minutes=minutes)
        out.append({
            "datasetId": dataset_id,
            "startTime": s.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "endTime": e.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "value": value_fn(i),
        })
    return out


def _page(rows, page=1, last=1):
    return {
        "data": rows,
        "pagination": {
            "total": len(rows), "lastPage": last, "prevPage": page - 1 or None,
            "nextPage": page + 1 if page < last else None, "perPage": 20000,
            "currentPage": page, "from": 0, "to": len(rows),
        },
    }


class FakeSession:
    """Routes GETs by dataset id to a handler(params) -> FakeResponse."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        dataset_id = int(url.rstrip("/").split("/")[-2])
        self.calls.append((dataset_id, dict(params), dict(headers or {})))
        handler = self.handlers.get(dataset_id)
        if handler is None:
            return FakeResponse(payload=_page([]))
        return handler(params)


START = datetime(2024, 1, 1, 0, tzinfo=timezone.utc)
END = datetime(2024, 1, 1, 2, tzinfo=timezone.utc)


def _client(session):
    return FingridClient(api_key="k", session=session, min_interval_s=0, sleep=lambda s: None)


def test_is_available(monkeypatch):
    monkeypatch.delenv("FINGRID_API_KEY", raising=False)
    assert not FingridClient().is_available()
    assert FingridClient(api_key="x").is_available()
    monkeypatch.setenv("FINGRID_API_KEY", "env")
    assert FingridClient().is_available()


def test_no_key_returns_empty_frame(monkeypatch):
    monkeypatch.delenv("FINGRID_API_KEY", raising=False)
    df = FingridClient().fetch_drivers(START, END)
    assert list(df.columns) == ["timestamp", "series", "value"]
    assert df.empty


def test_fetch_drivers_resamples_to_hourly_utc():
    handlers = {
        # 15-min wind forecast: hour 0 -> 100..103, hour 1 -> 104..107
        245: lambda p: FakeResponse(payload=_page(_points(245, START, 15, 8, lambda i: 100 + i))),
        # 3-min nuclear: 40 points over two hours, constant per hour
        188: lambda p: FakeResponse(payload=_page(
            _points(188, START, 3, 40, lambda i: 4000.0 if i < 20 else 4400.0))),
        # hourly capacity
        268: lambda p: FakeResponse(payload=_page(_points(268, START, 60, 2, lambda i: 8000 + i))),
    }
    session = FakeSession(handlers)
    df = _client(session).fetch_drivers(START, END)

    assert list(df.columns) == ["timestamp", "series", "value"]
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert set(df["series"]) == {S.WIND_FORECAST_MW, S.NUCLEAR_ACTUAL_MW, S.WIND_CAPACITY_MW}
    wide = df.pivot(index="timestamp", columns="series", values="value")
    assert list(wide.index) == [pd.Timestamp(START), pd.Timestamp(START) + pd.Timedelta(hours=1)]
    assert wide[S.WIND_FORECAST_MW].tolist() == [101.5, 105.5]
    assert wide[S.NUCLEAR_ACTUAL_MW].tolist() == [4000.0, 4400.0]
    assert wide[S.WIND_CAPACITY_MW].tolist() == [8000.0, 8001.0]

    # all datasets requested, with key header and documented params
    assert {c[0] for c in session.calls} == set(fg.DATASETS.values())
    _, params, headers = session.calls[0]
    assert headers["x-api-key"] == "k"
    assert params["startTime"] == "2024-01-01T00:00:00Z"
    assert params["endTime"] == "2024-01-01T02:00:00Z"
    assert params["format"] == "json"
    assert params["sortOrder"] == "asc"


def test_pagination_is_followed():
    rows = _points(124, START, 15, 8, lambda i: float(i))

    def handler(params):
        page = int(params["page"])
        chunk = rows[(page - 1) * 3: page * 3]
        return FakeResponse(payload=_page(chunk, page=page, last=3))

    session = FakeSession({124: handler})
    df = _client(session).fetch_drivers(START, END)
    pages = [c[1]["page"] for c in session.calls if c[0] == 124]
    assert pages == [1, 2, 3]
    s = df[df.series == S.CONSUMPTION_ACTUAL_MW].set_index("timestamp")["value"]
    assert s.tolist() == [1.5, 5.5]


def test_pagination_when_only_first_page_has_last_page():
    # Real Fingrid behaviour: pages after the first only carry currentPage/perPage.
    rows = _points(124, START, 15, 8, lambda i: float(i))

    def handler(params):
        page = int(params["page"])
        chunk = rows[(page - 1) * 3: page * 3]
        payload = _page(chunk, page=page, last=3)
        if page > 1:
            payload["pagination"] = {"currentPage": page, "perPage": 20000}
        return FakeResponse(payload=payload)

    session = FakeSession({124: handler})
    df = _client(session).fetch_drivers(START, END)
    pages = [c[1]["page"] for c in session.calls if c[0] == 124]
    assert pages == [1, 2, 3]
    s = df[df.series == S.CONSUMPTION_ACTUAL_MW].set_index("timestamp")["value"]
    assert s.tolist() == [1.5, 5.5]


def test_failing_dataset_is_skipped():
    handlers = {
        245: lambda p: FakeResponse(status_code=403, payload={"message": "forbidden"}),
        75: lambda p: FakeResponse(payload=_page(_points(75, START, 15, 8, lambda i: 50.0))),
    }
    df = _client(FakeSession(handlers)).fetch_drivers(START, END)
    assert set(df["series"]) == {S.WIND_ACTUAL_MW}
    assert df["value"].tolist() == [50.0, 50.0]


def test_429_is_retried_with_backoff():
    calls = {"n": 0}
    sleeps = []

    def handler(params):
        calls["n"] += 1
        if calls["n"] == 1:
            return FakeResponse(status_code=429, headers={"Retry-After": "3"})
        return FakeResponse(payload=_page(_points(245, START, 60, 2, lambda i: 10.0)))

    client = FingridClient(api_key="k", session=FakeSession({245: handler}),
                           min_interval_s=0, sleep=sleeps.append)
    df = client.fetch_forecasts(START, END)
    assert calls["n"] == 2
    assert sleeps == [3.0]
    assert set(df["series"]) == {S.WIND_FORECAST_MW}


def test_fetch_forecasts_only_forecast_series():
    session = FakeSession({})
    _client(session).fetch_forecasts(START, END)
    requested = {c[0] for c in session.calls}
    expected = {fg.DATASETS[n] for n in fg.DATASETS if "_forecast_" in n}
    assert requested == expected
    assert fg.DATASETS[S.WIND_ACTUAL_MW] not in requested


def test_long_range_is_chunked():
    session = FakeSession({})
    end = datetime(2024, 8, 1, tzinfo=timezone.utc)  # 213 days -> 2 chunks of <= 186 days
    _client(session).fetch_forecasts(START, end)
    windows = [(c[1]["startTime"], c[1]["endTime"]) for c in session.calls if c[0] == 245]
    assert windows == [
        ("2024-01-01T00:00:00Z", "2024-07-05T00:00:00Z"),
        ("2024-07-05T00:00:00Z", "2024-08-01T00:00:00Z"),
    ]


def test_output_limited_to_range_and_deduplicated():
    # chunk boundary overlap -> same point twice; also a point outside [start, end)
    rows = _points(166, START, 60, 3, lambda i: 7.0)
    rows.append(dict(rows[0]))
    session = FakeSession({166: lambda p: FakeResponse(payload=_page(rows))})
    df = _client(session).fetch_forecasts(START, END)
    assert len(df) == 2
    assert df["timestamp"].max() < pd.Timestamp(END)
