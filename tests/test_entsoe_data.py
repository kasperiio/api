"""Tests for app.forecasting.sources.entsoe_data (mocked HTTP, no network)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from app.forecasting import series as S
from app.forecasting.sources import entsoe_data as E
from app.forecasting.sources.entsoe_data import EntsoeDataClient, EntsoeNoData, parse_timeseries

PUB_NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"
GL_NS = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"
ACK_NS = "urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0"


def _period(start, end, res, points, tag):
    pts = "".join(
        "<Point><position>%d</position><%s>%s</%s></Point>" % (p, tag, v, tag) for p, v in points
    )
    return (
        "<Period><timeInterval><start>%s</start><end>%s</end></timeInterval>"
        "<resolution>%s</resolution>%s</Period>" % (start, end, res, pts)
    )


def _ts(periods, curve="A01", psr=None):
    psr_xml = "<MktPSRType><psrType>%s</psrType></MktPSRType>" % psr if psr else ""
    return "<TimeSeries><mRID>1</mRID><curveType>%s</curveType>%s%s</TimeSeries>" % (
        curve, psr_xml, "".join(periods))


def _doc(timeseries, ns=PUB_NS, root="Publication_MarketDocument"):
    return '<?xml version="1.0" encoding="utf-8"?><%s xmlns="%s"><mRID>x</mRID>%s</%s>' % (
        root, ns, "".join(timeseries), root)


ACK_XML = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<Acknowledgement_MarketDocument xmlns="%s"><mRID>a</mRID>'
    "<Reason><code>999</code><text>No matching data found for Data item Day-ahead Prices "
    "[12.1.D] (10YFI-1--------U, 10YFI-1--------U) and interval 2026-01-01T00:00Z/2026-01-02T00:00Z."
    "</text></Reason></Acknowledgement_MarketDocument>" % ACK_NS
)

# 24 hourly prices, PT60M, curve A01
PT60_XML = _doc([_ts([_period("2025-09-01T22:00Z", "2025-09-02T22:00Z", "PT60M",
                              [(i, 50.0 + i) for i in range(1, 25)], "price.amount")])])

# 2 hours of 15-min prices, PT15M, curve A03 with gaps (positions 2,3 and 6,7,8 missing)
PT15_A03_XML = _doc([_ts([_period("2025-10-02T22:00Z", "2025-10-03T00:00Z", "PT15M",
                                  [(1, 10.0), (4, 40.0), (5, -5.0)], "price.amount")],
                         curve="A03")])


def _resp(text, status=200):
    r = MagicMock()
    r.text = text
    r.status_code = status
    return r


def _client(responses):
    session = MagicMock()
    if callable(responses):
        session.get.side_effect = responses
    else:
        session.get.side_effect = list(responses)
    c = EntsoeDataClient(api_key="test-key", session=session)
    c.MIN_REQUEST_INTERVAL = 0
    return c, session


UTC = timezone.utc


# --- parsing ---------------------------------------------------------------

def test_parse_pt60m():
    parsed = parse_timeseries(PT60_XML, "price.amount")
    assert len(parsed) == 1
    ts = parsed[0]
    assert ts["resolution_minutes"] == 60
    assert len(ts["points"]) == 24
    assert ts["points"][0] == (pd.Timestamp("2025-09-01T22:00Z"), 51.0)
    assert ts["points"][-1][0] == pd.Timestamp("2025-09-02T21:00Z")


def test_parse_pt15m_a03_fills_gaps_until_period_end():
    parsed = parse_timeseries(PT15_A03_XML, "price.amount")
    pts = parsed[0]["points"]
    assert parsed[0]["resolution_minutes"] == 15
    assert len(pts) == 8  # 2h / 15min, trailing positions repeat the last value
    vals = [v for _, v in pts]
    assert vals == [10.0, 10.0, 10.0, 40.0, -5.0, -5.0, -5.0, -5.0]
    assert [t.minute for t, _ in pts[:4]] == [0, 15, 30, 45]
    assert pts[-1][0] == pd.Timestamp("2025-10-02T23:45Z")


def test_parse_ack_raises_no_data():
    with pytest.raises(EntsoeNoData, match="No matching data"):
        parse_timeseries(ACK_XML, "price.amount")


def test_parse_multi_timeseries_and_psr():
    xml = _doc([
        _ts([_period("2026-01-01T00:00Z", "2026-01-01T01:00Z", "PT15M",
                     [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)], "quantity")], psr="B16"),
        _ts([_period("2026-01-01T00:00Z", "2026-01-01T01:00Z", "PT60M", [(1, 100.0)], "quantity")],
            psr="B19"),
    ], ns=GL_NS, root="GL_MarketDocument")
    parsed = parse_timeseries(xml, "quantity")
    assert [p["attrs"]["psrType"] for p in parsed] == ["B16", "B19"]
    assert [p["resolution_minutes"] for p in parsed] == [15, 60]


def test_parse_weekly_resolution():
    xml = _doc([_ts([_period("2026-01-04T23:00Z", "2026-01-18T23:00Z", "P7D",
                             [(1, 1000.0), (2, 900.0)], "quantity")])],
               ns=GL_NS, root="GL_MarketDocument")
    parsed = parse_timeseries(xml, "quantity")
    assert parsed[0]["resolution_minutes"] == 7 * 24 * 60
    assert parsed[0]["points"][1][0] == pd.Timestamp("2026-01-11T23:00Z")


# --- fetch_prices -------------------------------------------------------------

def test_fetch_prices_pt60_no_vat_utc():
    c, session = _client([_resp(PT60_XML)])
    df = c.fetch_prices(datetime(2025, 9, 1, 22, tzinfo=UTC), datetime(2025, 9, 2, 22, tzinfo=UTC),
                        zones=("FI",))
    assert list(df.columns) == ["timestamp", "zone", "price_eur_mwh", "resolution_minutes"]
    assert len(df) == 24
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert (df["zone"] == "FI").all()
    assert (df["resolution_minutes"] == 60).all()
    assert df["price_eur_mwh"].iloc[0] == 51.0  # raw EUR/MWh, no VAT / unit conversion
    params = session.get.call_args.kwargs["params"]
    assert params["documentType"] == "A44"
    assert params["in_Domain"] == params["out_Domain"] == "10YFI-1--------U"
    assert params["periodStart"] == "202509012200"
    assert params["securityToken"] == "test-key"


def test_fetch_prices_pt15_a03_and_window_clip():
    c, _ = _client([_resp(PT15_A03_XML)])
    df = c.fetch_prices(datetime(2025, 10, 2, 22, 30, tzinfo=UTC),
                        datetime(2025, 10, 3, tzinfo=UTC), zones=("FI",))
    assert len(df) == 6
    assert (df["resolution_minutes"] == 15).all()
    assert df["timestamp"].iloc[0] == pd.Timestamp("2025-10-02T22:30Z")
    assert df["price_eur_mwh"].tolist() == [10.0, 40.0, -5.0, -5.0, -5.0, -5.0]


def test_fetch_prices_multiple_zones_and_no_data_zone():
    def handler(url, params, timeout):
        if params["in_Domain"] == E.ZONE_EIC["SE1"]:
            return _resp(ACK_XML, status=400)
        return _resp(PT60_XML)

    c, _ = _client(handler)
    df = c.fetch_prices(datetime(2025, 9, 1, 22, tzinfo=UTC), datetime(2025, 9, 2, 22, tzinfo=UTC),
                        zones=("FI", "SE1", "EE"))
    assert sorted(df["zone"].unique()) == ["EE", "FI"]
    assert len(df) == 48


def test_fetch_prices_prefers_15min_over_hourly_for_same_hour():
    xml = _doc([
        _ts([_period("2025-10-02T22:00Z", "2025-10-02T23:00Z", "PT60M", [(1, 99.0)], "price.amount")]),
        _ts([_period("2025-10-02T22:00Z", "2025-10-02T23:00Z", "PT15M",
                     [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0)], "price.amount")]),
    ])
    c, _ = _client([_resp(xml)])
    df = c.fetch_prices(datetime(2025, 10, 2, 22, tzinfo=UTC), datetime(2025, 10, 2, 23, tzinfo=UTC),
                        zones=("FI",))
    assert df["resolution_minutes"].tolist() == [15, 15, 15, 15]
    assert df["price_eur_mwh"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_fetch_prices_chunks_requests():
    c, session = _client(lambda url, params, timeout: _resp(ACK_XML, 400))
    c.chunk_days = 31
    df = c.fetch_prices(datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 4, 1, tzinfo=UTC),
                        zones=("FI",))
    assert df.empty
    assert session.get.call_count == 3  # 90 days / 31-day chunks


def test_fetch_prices_requires_tz():
    c, _ = _client([])
    with pytest.raises(ValueError):
        c.fetch_prices(datetime(2025, 1, 1), datetime(2025, 1, 2))


# --- fetch_drivers / fetch_forecasts -----------------------------------------

LOAD_XML = _doc([_ts([_period("2026-01-01T00:00Z", "2026-01-01T02:00Z", "PT15M",
                              [(1, 100.0), (2, 200.0), (3, 300.0), (4, 400.0), (5, 1000.0)],
                              "quantity")], curve="A03")], ns=GL_NS, root="GL_MarketDocument")

WIND_SOLAR_XML = _doc([
    _ts([_period("2026-01-01T00:00Z", "2026-01-01T02:00Z", "PT60M", [(1, 1000.0), (2, 1100.0)],
                 "quantity")], psr="B19"),
    _ts([_period("2026-01-01T00:00Z", "2026-01-01T02:00Z", "PT60M", [(1, 40.0), (2, 60.0)],
                 "quantity")], psr="B18"),
    _ts([_period("2026-01-01T00:00Z", "2026-01-01T02:00Z", "PT15M",
                 [(i, float(i)) for i in range(1, 9)], "quantity")], psr="B16"),
], ns=GL_NS, root="GL_MarketDocument")

RES_NO_XML = _doc([_ts([_period("2025-12-21T23:00Z", "2025-12-28T23:00Z", "P7D", [(1, 5e7)],
                                "quantity")])], ns=GL_NS, root="GL_MarketDocument")


def _driver_handler(fail_se=True):
    def handler(url, params, timeout):
        dt = params["documentType"]
        if dt == "A65":
            return _resp(LOAD_XML)
        if dt == "A69":
            return _resp(WIND_SOLAR_XML)
        if dt == "A72" and params["in_Domain"] == E.COUNTRY_EIC["NO"]:
            return _resp(RES_NO_XML)
        return _resp(ACK_XML, 400 if fail_se else 200)
    return handler


def test_fetch_drivers_hourly_and_series():
    c, session = _client(_driver_handler())
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 2, tzinfo=UTC)
    df = c.fetch_drivers(start, end)
    assert list(df.columns) == ["timestamp", "series", "value"]
    assert str(df["timestamp"].dt.tz) == "UTC"
    v = df.set_index(["series", "timestamp"])["value"]
    h0, h1 = pd.Timestamp("2026-01-01T00:00Z"), pd.Timestamp("2026-01-01T01:00Z")
    assert v[(S.LOAD_FORECAST_MW, h0)] == 250.0  # mean of 4 quarter-hours
    assert v[(S.LOAD_FORECAST_MW, h1)] == 1000.0  # A03 fill to period end
    assert v[(S.WIND_FORECAST_ENTSOE_MW, h0)] == 1040.0  # onshore + offshore
    assert v[(S.WIND_FORECAST_ENTSOE_MW, h1)] == 1160.0
    assert v[(S.SOLAR_FORECAST_ENTSOE_MW, h0)] == 2.5
    assert v[(S.SOLAR_FORECAST_ENTSOE_MW, h1)] == 6.5
    # Weekly NO reservoir forward-filled; SE failed and is absent
    assert v[(S.HYDRO_RESERVOIR_NO_MWH, h0)] == 5e7
    assert v[(S.HYDRO_RESERVOIR_NO_MWH, h1)] == 5e7
    assert S.HYDRO_RESERVOIR_SE_MWH not in set(df["series"])
    # Request params for each document type
    calls = {c_.kwargs["params"]["documentType"]: c_.kwargs["params"] for c_ in session.get.call_args_list}
    assert calls["A65"]["processType"] == "A01"
    assert calls["A65"]["outBiddingZone_Domain"] == "10YFI-1--------U"
    assert calls["A69"]["processType"] == "A01"
    assert calls["A72"]["processType"] == "A16"


def test_fetch_drivers_survives_http_error():
    def handler(url, params, timeout):
        if params["documentType"] == "A65":
            return _resp("<html>err</html>", 503)
        return _driver_handler()(url, params, timeout)

    c, _ = _client(handler)
    df = c.fetch_drivers(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 2, tzinfo=UTC))
    assert S.LOAD_FORECAST_MW not in set(df["series"])
    assert S.WIND_FORECAST_ENTSOE_MW in set(df["series"])


def test_fetch_forecasts_only_forecast_series():
    c, session = _client(_driver_handler())
    df = c.fetch_forecasts(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 2, tzinfo=UTC))
    assert set(df["series"]) == {S.LOAD_FORECAST_MW, S.WIND_FORECAST_ENTSOE_MW,
                                 S.SOLAR_FORECAST_ENTSOE_MW}
    assert all("_forecast_" in s for s in df["series"])
    assert "A72" not in {c_.kwargs["params"]["documentType"] for c_ in session.get.call_args_list}


def test_transfer_capacities_query_import_direction_and_daily_contract():
    c, session = _client(lambda *args, **kwargs: _resp(LOAD_XML))
    start, end = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, 2, tzinfo=UTC)
    frame = c.fetch_transfer_capacities(start, end)
    assert set(frame.series) == set(S.TRANSFER_CAPACITY_SERIES)
    for call in session.get.call_args_list:
        params = call.kwargs["params"]
        assert params["documentType"] == "A61"
        assert params["contract_MarketAgreement.type"] == "A01"
        assert params["in_Domain"] == E.ZONE_EIC["FI"]
        assert params["out_Domain"] in {E.ZONE_EIC[z] for z in ("EE", "SE1", "SE3")}


def test_weekly_ffill_limited_extension():
    parsed = parse_timeseries(RES_NO_XML, "quantity")
    t0, t1 = pd.Timestamp("2025-12-22T00:00Z"), pd.Timestamp("2026-03-01T00:00Z")
    df = E._weekly_ffill(parsed, S.HYDRO_RESERVOIR_NO_MWH, t0, t1)
    # week ends 2025-12-28T23:00Z, carried forward 14 more days
    assert df["timestamp"].max() == pd.Timestamp("2026-01-11T22:00Z")
    assert df["timestamp"].min() == t0


def test_is_available(monkeypatch):
    assert EntsoeDataClient(api_key="k", session=MagicMock()).is_available()
    monkeypatch.setattr(E.os, "getenv", lambda *a, **k: None)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    assert not EntsoeDataClient(session=MagicMock()).is_available()
