"""ENTSO-E Transparency client for forecasting data. Requires ENTSOE_API_KEY.

Endpoints used (https://web-api.tp.entsoe.eu/api):

- Day-ahead prices: documentType A44, in_Domain = out_Domain = zone EIC,
  contract_MarketAgreement.type A01. EUR/MWh, no VAT. Native resolution is
  kept (PT60M before 2025-10-01, PT15M after the SDAC 15-min MTU go-live).
- Load forecast day-ahead: documentType A65, processType A01,
  outBiddingZone_Domain = FI.
- Wind/solar day-ahead forecast: documentType A69, processType A01,
  in_Domain = FI, psrType B16 (solar), B18 (wind offshore), B19 (wind onshore).
- Hydro reservoir filling: documentType A72, processType A16, in_Domain =
  country EIC (NO / SE), weekly (P7D) MWh values.

All timestamps are tz-aware UTC.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.forecasting import series as S
from app.logging_config import logger

BASE_URL = "https://web-api.tp.entsoe.eu/api"

# Bidding zone EIC codes (day-ahead prices, load, generation forecasts)
ZONE_EIC: Dict[str, str] = {
    "FI": "10YFI-1--------U",
    "SE1": "10Y1001A1001A44P",
    "SE2": "10Y1001A1001A45N",
    "SE3": "10Y1001A1001A46L",
    "SE4": "10Y1001A1001A47J",
    "EE": "10Y1001A1001A39I",
}

# Country EIC codes (hydro reservoir filling is published per country)
COUNTRY_EIC: Dict[str, str] = {
    "NO": "10YNO-0--------C",
    "SE": "10YSE-1--------K",
}

PSR_SOLAR = "B16"
PSR_WIND_OFFSHORE = "B18"
PSR_WIND_ONSHORE = "B19"

FORECAST_SERIES = (S.LOAD_FORECAST_MW, S.WIND_FORECAST_ENTSOE_MW, S.SOLAR_FORECAST_ENTSOE_MW)

PRICE_COLUMNS = ["timestamp", "zone", "price_eur_mwh", "resolution_minutes"]
DRIVER_COLUMNS = ["timestamp", "series", "value"]


class EntsoeNoData(Exception):
    """Raised for an Acknowledgement document ('No matching data found')."""


class EntsoeError(Exception):
    """Raised for HTTP / parse failures."""


# ---------------------------------------------------------------------------
# Parsing helpers (module-level so they can be unit-tested without HTTP)
# ---------------------------------------------------------------------------

def _parse_duration(text: str) -> pd.Timedelta:
    """Parse ENTSO-E ISO-8601 resolutions such as PT15M, PT60M, PT1H, P1D, P7D."""
    text = text.strip().upper()
    if text in ("P1M", "P1Y"):
        raise ValueError("Calendar resolution %s not supported" % text)
    return pd.Timedelta(text)


def _parse_ts(text: str) -> pd.Timestamp:
    ts = pd.Timestamp(text.strip())
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _ns(root: ET.Element) -> Dict[str, str]:
    if root.tag.startswith("{"):
        return {"ns": root.tag[1:].split("}")[0]}
    return {"ns": ""}


def _find_text(el: ET.Element, path: str, ns: Dict[str, str]) -> Optional[str]:
    found = el.find(path, ns)
    return found.text if found is not None and found.text is not None else None


def _check_ack(root: ET.Element) -> None:
    if "Acknowledgement_MarketDocument" in root.tag:
        texts = [t.text for t in root.iter() if t.tag.endswith("}text") or t.tag == "text"]
        msg = "; ".join(t for t in texts if t) or "Acknowledgement document"
        raise EntsoeNoData(msg)


def parse_timeseries(xml_text: str, value_tag: str) -> List[dict]:
    """Parse an ENTSO-E market document into a list of TimeSeries dicts.

    Each dict: ``{"attrs": {...}, "points": [(timestamp, value), ...],
    "resolution_minutes": int}``. Curve type A03 (variable sized block) is
    expanded: missing positions repeat the previous point until the next given
    position or the period end. Raises EntsoeNoData for acknowledgement docs.
    """
    try:
        root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    except ET.ParseError as e:
        raise EntsoeError("Failed to parse XML: %s" % e)
    _check_ack(root)
    ns = _ns(root)

    out = []
    for ts_el in root.findall("ns:TimeSeries", ns):
        curve = _find_text(ts_el, "ns:curveType", ns) or "A01"
        attrs = {
            "curveType": curve,
            "psrType": _find_text(ts_el, "ns:MktPSRType/ns:psrType", ns),
            "classification": _find_text(
                ts_el, "ns:classificationSequence_AttributeInstanceComponent.position", ns
            ),
        }
        for period in ts_el.findall("ns:Period", ns):
            start_txt = _find_text(period, "ns:timeInterval/ns:start", ns)
            end_txt = _find_text(period, "ns:timeInterval/ns:end", ns)
            res_txt = _find_text(period, "ns:resolution", ns)
            if not (start_txt and end_txt and res_txt):
                continue
            p_start, p_end = _parse_ts(start_txt), _parse_ts(end_txt)
            res = _parse_duration(res_txt)
            n_slots = int((p_end - p_start) / res)

            given: Dict[int, float] = {}
            for pt in period.findall("ns:Point", ns):
                pos_txt = _find_text(pt, "ns:position", ns)
                val_txt = _find_text(pt, "ns:" + value_tag, ns)
                if pos_txt is None or val_txt is None:
                    continue
                try:
                    given[int(pos_txt)] = float(val_txt)
                except ValueError:
                    continue
            if not given:
                continue

            points: List[Tuple[pd.Timestamp, float]] = []
            if curve == "A03":
                last = None
                for pos in range(1, max(n_slots, max(given)) + 1):
                    if pos in given:
                        last = given[pos]
                    if last is None:
                        continue
                    points.append((p_start + (pos - 1) * res, last))
            else:
                for pos in sorted(given):
                    points.append((p_start + (pos - 1) * res, given[pos]))

            out.append({
                "attrs": attrs,
                "points": points,
                "resolution_minutes": int(res.total_seconds() // 60),
            })
    return out


def _price_rows(parsed: List[dict], zone: str) -> pd.DataFrame:
    rows = []
    for ts in parsed:
        # Prefer the primary classification sequence (position 1 / absent)
        prio = 0 if ts["attrs"].get("classification") in (None, "1") else 1
        for t, v in ts["points"]:
            rows.append((t, zone, v, ts["resolution_minutes"], prio))
    if not rows:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    df = pd.DataFrame(rows, columns=PRICE_COLUMNS + ["_prio"])
    df = (df.sort_values(["_prio", "resolution_minutes", "timestamp"])
            .drop_duplicates(subset=["timestamp"], keep="first"))
    # If an hour has both 15-min and 60-min data, keep only the finer one.
    hour = df["timestamp"].dt.floor("h")
    finest = df.groupby(hour)["resolution_minutes"].transform("min")
    df = df[df["resolution_minutes"] == finest]
    return df.drop(columns="_prio").sort_values("timestamp").reset_index(drop=True)


def _to_hourly(points: List[Tuple[pd.Timestamp, float]], series: str) -> pd.DataFrame:
    if not points:
        return pd.DataFrame(columns=DRIVER_COLUMNS)
    s = pd.Series([v for _, v in points], index=pd.DatetimeIndex([t for t, _ in points]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    hourly = s.resample("h").mean().dropna()
    df = hourly.rename("value").rename_axis("timestamp").reset_index()
    df["series"] = series
    return df[DRIVER_COLUMNS]


def _sum_series(parsed: List[dict], psr_types: Tuple[str, ...]) -> List[Tuple[pd.Timestamp, float]]:
    """Sum native-resolution points across TimeSeries with the given psrTypes.

    Each series is resampled hourly first so that offshore/onshore series with
    different resolutions add up correctly.
    """
    parts = []
    for ts in parsed:
        if ts["attrs"].get("psrType") not in psr_types or not ts["points"]:
            continue
        s = pd.Series([v for _, v in ts["points"]],
                      index=pd.DatetimeIndex([t for t, _ in ts["points"]]))
        parts.append((ts["attrs"]["psrType"], s))
    if not parts:
        return []
    by_psr: Dict[str, pd.Series] = {}
    for psr, s in parts:
        hourly = s.resample("h").mean()
        by_psr[psr] = hourly if psr not in by_psr else hourly.combine_first(by_psr[psr])
    total = pd.concat(by_psr.values(), axis=1).sum(axis=1, min_count=1).dropna()
    return list(zip(total.index, total.values))


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class EntsoeDataClient:
    MIN_REQUEST_INTERVAL = 60.0 / 350  # stay below the 400 requests/minute limit

    def __init__(self, api_key: str = None, session: Optional[requests.Session] = None,
                 chunk_days: int = 31, max_workers: int = 4):
        if api_key is None:
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:  # pragma: no cover
                pass
            api_key = os.getenv("ENTSOE_API_KEY")
        self.api_key = api_key or None
        self.chunk_days = chunk_days
        # ENTSO-E answers slowly (~10 s per request) but allows 400 requests/minute,
        # so independent zone/series requests run concurrently.
        self.max_workers = max(1, max_workers)
        self._last_request = 0.0
        self._throttle_lock = threading.Lock()
        if session is None:
            session = requests.Session()
            retry = Retry(total=3, backoff_factor=1.0,
                          status_forcelist=[429, 500, 502, 503, 504],
                          allowed_methods=["GET"])
            session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session = session

    def is_available(self) -> bool:
        return bool(self.api_key)

    # -- HTTP ---------------------------------------------------------------

    @staticmethod
    def _fmt(ts: pd.Timestamp) -> str:
        return ts.strftime("%Y%m%d%H%M")

    def _throttle(self) -> None:
        # Reserve the next request slot under the lock, sleep outside it.
        with self._throttle_lock:
            now = time.monotonic()
            slot = max(now, self._last_request + self.MIN_REQUEST_INTERVAL)
            self._last_request = slot
        if slot > now:
            time.sleep(slot - now)

    def _map(self, fn, items) -> list:
        """``[fn(item) for item in items]``, run concurrently, results in order."""
        items = list(items)
        if self.max_workers == 1 or len(items) <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(items))) as pool:
            return list(pool.map(fn, items))

    def _get(self, params: Dict[str, str]) -> str:
        if not self.api_key:
            raise EntsoeError("ENTSOE_API_KEY not configured")
        full = dict(params)
        full["securityToken"] = self.api_key
        self._throttle()
        try:
            resp = self.session.get(BASE_URL, params=full, timeout=60)
        except requests.exceptions.RequestException as e:
            raise EntsoeError("ENTSO-E request failed: %s" % type(e).__name__)
        if resp.status_code == 401:
            raise EntsoeError("Invalid ENTSO-E API key (401)")
        # "No matching data" comes back as an Acknowledgement doc, often with 400
        if "Acknowledgement_MarketDocument" in resp.text:
            parse_timeseries(resp.text, "quantity")  # raises EntsoeNoData
        if resp.status_code != 200:
            raise EntsoeError("ENTSO-E HTTP %s" % resp.status_code)
        return resp.text

    def _chunks(self, start: pd.Timestamp, end: pd.Timestamp):
        cur = start
        step = pd.Timedelta(days=self.chunk_days)
        while cur < end:
            nxt = min(cur + step, end)
            yield cur, nxt
            cur = nxt

    def _query(self, params: Dict[str, str], value_tag: str,
               start: pd.Timestamp, end: pd.Timestamp) -> List[dict]:
        """Run a chunked query; chunks with no data are skipped silently."""
        parsed: List[dict] = []
        for c_start, c_end in self._chunks(start, end):
            p = dict(params, periodStart=self._fmt(c_start), periodEnd=self._fmt(c_end))
            try:
                parsed.extend(parse_timeseries(self._get(p), value_tag))
            except EntsoeNoData as e:
                logger.debug("ENTSO-E no data for %s %s-%s: %s",
                             params.get("documentType"), c_start, c_end, e)
        return parsed

    # -- Public API ---------------------------------------------------------

    def fetch_prices(self, start: datetime, end: datetime, zones=("FI", "SE1", "SE3", "EE")) -> pd.DataFrame:
        """Price frame (timestamp, zone, price_eur_mwh, resolution_minutes) for [start, end)."""
        t0, t1 = _utc_bounds(start, end)

        def fetch_zone(zone: str) -> Optional[pd.DataFrame]:
            eic = ZONE_EIC[zone]
            params = {
                "documentType": "A44",
                "in_Domain": eic,
                "out_Domain": eic,
                "contract_MarketAgreement.type": "A01",
            }
            try:
                parsed = self._query(params, "price.amount", t0, t1)
            except EntsoeError as e:
                logger.warning("ENTSO-E prices for %s failed: %s", zone, e)
                return None
            df = _price_rows(parsed, zone)
            return df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]

        frames = [df for df in self._map(fetch_zone, zones) if df is not None and not df.empty]
        if not frames:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        out["price_eur_mwh"] = out["price_eur_mwh"].astype(float)
        out["resolution_minutes"] = out["resolution_minutes"].astype(int)
        return out[PRICE_COLUMNS].reset_index(drop=True)

    def fetch_drivers(self, start: datetime, end: datetime, include_reservoirs: bool = True) -> pd.DataFrame:
        """Driver frame (timestamp, series, value) of the ENTSO-E series in app.forecasting.series."""
        t0, t1 = _utc_bounds(start, end)
        tasks = [lambda: [self._load_forecast(t0, t1)], lambda: self._wind_solar_forecast(t0, t1)]
        if include_reservoirs:
            for country, name in (("NO", S.HYDRO_RESERVOIR_NO_MWH), ("SE", S.HYDRO_RESERVOIR_SE_MWH)):
                tasks.append(lambda c=country, n=name: [self._reservoir(c, n, t0, t1)])
        frames = [f for result in self._map(lambda task: task(), tasks) for f in result]
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return pd.DataFrame(columns=DRIVER_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        out = out[(out["timestamp"] >= t0) & (out["timestamp"] < t1)]
        out["value"] = out["value"].astype(float)
        return out.sort_values(["series", "timestamp"]).reset_index(drop=True)[DRIVER_COLUMNS]

    def fetch_forecasts(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Driver frame of only the ``*_forecast_*`` series (used for hourly snapshots)."""
        df = self.fetch_drivers(start, end, include_reservoirs=False)
        return df[df["series"].isin(FORECAST_SERIES)].reset_index(drop=True)

    # -- Individual driver series -------------------------------------------

    def _load_forecast(self, t0, t1) -> pd.DataFrame:
        params = {"documentType": "A65", "processType": "A01",
                  "outBiddingZone_Domain": ZONE_EIC["FI"]}
        try:
            parsed = self._query(params, "quantity", t0, t1)
        except EntsoeError as e:
            logger.warning("ENTSO-E load forecast failed: %s", e)
            return pd.DataFrame(columns=DRIVER_COLUMNS)
        points = [p for ts in parsed for p in ts["points"]]
        if not points:
            logger.warning("ENTSO-E load forecast: no data for %s - %s", t0, t1)
        return _to_hourly(points, S.LOAD_FORECAST_MW)

    def _wind_solar_forecast(self, t0, t1) -> List[pd.DataFrame]:
        params = {"documentType": "A69", "processType": "A01", "in_Domain": ZONE_EIC["FI"]}
        try:
            parsed = self._query(params, "quantity", t0, t1)
        except EntsoeError as e:
            logger.warning("ENTSO-E wind/solar forecast failed: %s", e)
            return []
        out = []
        for name, psrs in ((S.WIND_FORECAST_ENTSOE_MW, (PSR_WIND_ONSHORE, PSR_WIND_OFFSHORE)),
                           (S.SOLAR_FORECAST_ENTSOE_MW, (PSR_SOLAR,))):
            points = _sum_series(parsed, psrs)
            if not points:
                logger.warning("ENTSO-E %s: no data for %s - %s", name, t0, t1)
            out.append(_to_hourly(points, name))
        return out

    def _reservoir(self, country: str, name: str, t0, t1) -> pd.DataFrame:
        # Look back so the first hours of the window have a weekly value to forward-fill.
        params = {"documentType": "A72", "processType": "A16", "in_Domain": COUNTRY_EIC[country]}
        try:
            parsed = self._query(params, "quantity", t0 - pd.Timedelta(days=21), t1)
        except EntsoeError as e:
            logger.warning("ENTSO-E reservoir %s failed: %s", country, e)
            return pd.DataFrame(columns=DRIVER_COLUMNS)
        return _weekly_ffill(parsed, name, t0, t1)


def _weekly_ffill(parsed: List[dict], name: str, t0: pd.Timestamp, t1: pd.Timestamp,
                  max_extend: pd.Timedelta = pd.Timedelta(days=14)) -> pd.DataFrame:
    """Expand weekly points to hourly by forward fill.

    Each weekly value covers its period; after the last published week the
    value is carried forward at most ``max_extend`` beyond that period.
    """
    pts = sorted({t: v for ts in parsed for t, v in ts["points"]}.items())
    if not pts:
        logger.warning("ENTSO-E %s: no data for %s - %s", name, t0, t1)
        return pd.DataFrame(columns=DRIVER_COLUMNS)
    s = pd.Series([v for _, v in pts], index=pd.DatetimeIndex([t for t, _ in pts]))
    stop = min(t1, s.index[-1] + pd.Timedelta(days=7) + max_extend)
    idx = pd.date_range(s.index[0], stop, freq="h", inclusive="left", tz="UTC")
    hourly = s.reindex(idx.union(s.index)).ffill().reindex(idx)
    hourly = hourly[(hourly.index >= t0)].dropna()
    df = hourly.rename("value").rename_axis("timestamp").reset_index()
    df["series"] = name
    return df[DRIVER_COLUMNS]


def _utc_bounds(start: datetime, end: datetime) -> Tuple[pd.Timestamp, pd.Timestamp]:
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
    if t0.tzinfo is None or t1.tzinfo is None:
        raise ValueError("start/end must be timezone-aware")
    t0, t1 = t0.tz_convert("UTC"), t1.tz_convert("UTC")
    if t0 >= t1:
        raise ValueError("start must be before end")
    return t0, t1
