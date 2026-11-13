"""Fingrid Open Data client (https://data.fingrid.fi). Requires FINGRID_API_KEY.

API: ``GET https://data.fingrid.fi/api/datasets/{id}/data`` with header
``x-api-key``. Query params: startTime/endTime (``YYYY-MM-DDTHH:MM:SSZ``),
format=json, page, pageSize, sortBy, sortOrder. Response::

    {"data": [{"datasetId": 245, "startTime": "...Z", "endTime": "...Z", "value": 1.0}],
     "pagination": {"total": .., "lastPage": .., "nextPage": .., "currentPage": .., ...}}

Rate limits (per key): 10 requests/minute, 10 000 requests/24 h. Requests are
spaced and 429 responses retried with backoff.

Dataset IDs (verified from data.fingrid.fi dataset pages, 2026-11):

====  ===========================================================  =========================
 ID   Dataset                                                      Native resolution
====  ===========================================================  =========================
245   Wind power generation forecast - updated every 15 minutes    15 min (1 h before 2023-05-31)
 75   Wind power generation - 15 min data                          15 min (1 h before 2023-06-13)
268   Total production capacity used in the wind power forecast    1 h (updated ~weekly)
248   Solar power generation forecast - updated every 15 minutes   15 min (1 h before 2023-05-31)
166   Electricity consumption forecast - updated every 15 minutes  15 min (5 min before 2024-04-21), 72 h horizon
124   Electricity consumption in Finland                           15 min
241   Electricity production prediction - updated every 15 min     15 min (1 h before 2023-06-03)
188   Nuclear power production - real-time data                    3 min
191   Hydro power production - real-time data                      3 min
194   Net import/export of electricity - real-time data            3 min
====  ===========================================================  =========================

Everything is resampled to the hourly UTC mean (MW == MWh/h for hourly means).
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from app.forecasting import series as S
from app.logging_config import logger

# series name -> Fingrid dataset id
DATASETS: Dict[str, int] = {
    S.WIND_FORECAST_MW: 245,
    S.WIND_ACTUAL_MW: 75,
    S.WIND_CAPACITY_MW: 268,
    S.SOLAR_FORECAST_MW: 248,
    S.CONSUMPTION_FORECAST_MW: 166,
    S.CONSUMPTION_ACTUAL_MW: 124,
    S.PRODUCTION_FORECAST_MW: 241,
    S.NUCLEAR_ACTUAL_MW: 188,
    S.HYDRO_ACTUAL_MW: 191,
    S.NET_IMPORT_ACTUAL_MW: 194,
}

FORECAST_SERIES = [name for name in DATASETS if "_forecast_" in name]

COLUMNS = ["timestamp", "series", "value"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series([], dtype="datetime64[ns, UTC]"),
        "series": pd.Series([], dtype=object),
        "value": pd.Series([], dtype=float),
    })


def _to_utc(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


class FingridClient:
    BASE_URL = "https://data.fingrid.fi/api/datasets/{dataset_id}/data"
    PAGE_SIZE = 20000
    CHUNK_DAYS = 186  # one 6-month backfill window per request; pagination handles the size
    MIN_INTERVAL_S = 6.5  # 10 requests/minute per key
    MAX_RETRIES = 5
    TIMEOUT = (10, 60)

    def __init__(
        self,
        api_key: str = None,
        session: Optional[requests.Session] = None,
        min_interval_s: Optional[float] = None,
        backoff_s: float = 15.0,
        sleep=time.sleep,
    ):
        self.api_key = api_key or os.getenv("FINGRID_API_KEY")
        self.session = session or requests.Session()
        self.min_interval_s = self.MIN_INTERVAL_S if min_interval_s is None else min_interval_s
        self.backoff_s = backoff_s
        self._sleep = sleep
        self._last_request = 0.0

    def is_available(self) -> bool:
        return bool(self.api_key)

    def fetch_drivers(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Driver frame (timestamp, series, value) at hourly resolution for [start, end).
        Covers the Fingrid series in app.forecasting.series."""
        return self._fetch(list(DATASETS), start, end)

    def fetch_forecasts(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Driver frame of only the ``*_forecast_*`` series (used for hourly snapshots)."""
        return self._fetch(FORECAST_SERIES, start, end)

    # ------------------------------------------------------------------ internals

    def _fetch(self, names: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        if not self.is_available():
            logger.warning("Fingrid: FINGRID_API_KEY not set, skipping fetch")
            return _empty()
        start_ts, end_ts = _to_utc(start).floor("h"), _to_utc(end)
        if end_ts <= start_ts:
            return _empty()

        frames = []
        for name in names:
            dataset_id = DATASETS[name]
            try:
                raw = self._fetch_dataset(dataset_id, start_ts, end_ts)
            except Exception as e:  # skip a failing dataset, keep the rest
                logger.warning(f"Fingrid: dataset {dataset_id} ({name}) failed, skipping: {e}")
                continue
            hourly = self._to_hourly(raw, start_ts, end_ts)
            if hourly.empty:
                logger.info(f"Fingrid: dataset {dataset_id} ({name}) returned no data")
                continue
            frames.append(pd.DataFrame({
                "timestamp": hourly.index,
                "series": name,
                "value": hourly.values.astype(float),
            }))
        if not frames:
            return _empty()
        return pd.concat(frames, ignore_index=True)[COLUMNS]

    def _fetch_dataset(
        self, dataset_id: int, start: pd.Timestamp, end: pd.Timestamp
    ) -> List[dict]:
        rows: List[dict] = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + pd.Timedelta(days=self.CHUNK_DAYS), end)
            page = 1
            last_page = None
            while True:
                payload = self._get(dataset_id, {
                    "startTime": _fmt(chunk_start),
                    "endTime": _fmt(chunk_end),
                    "format": "json",
                    "page": page,
                    "pageSize": self.PAGE_SIZE,
                    "sortBy": "startTime",
                    "sortOrder": "asc",
                })
                data = payload.get("data") or []
                rows.extend(data)
                # Only the first page carries lastPage/nextPage; later pages just
                # have currentPage, so remember lastPage and fall back to "full page".
                pagination = payload.get("pagination") or {}
                if pagination.get("lastPage") is not None:
                    last_page = int(pagination["lastPage"])
                if last_page is not None:
                    more = page < last_page
                else:
                    more = bool(pagination.get("nextPage")) or len(data) >= self.PAGE_SIZE
                if not more or not data:
                    break
                page += 1
            chunk_start = chunk_end
        return rows

    def _get(self, dataset_id: int, params: dict) -> dict:
        url = self.BASE_URL.format(dataset_id=dataset_id)
        headers = {"x-api-key": self.api_key, "Accept": "application/json"}
        for attempt in range(self.MAX_RETRIES + 1):
            self._throttle()
            resp = self.session.get(url, params=params, headers=headers, timeout=self.TIMEOUT)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= self.MAX_RETRIES:
                    resp.raise_for_status()
                retry_after = resp.headers.get("Retry-After") if resp.headers else None
                try:
                    wait = float(retry_after) if retry_after else self.backoff_s * (2 ** attempt)
                except ValueError:
                    wait = self.backoff_s * (2 ** attempt)
                logger.info(
                    f"Fingrid: HTTP {resp.status_code} for dataset {dataset_id}, "
                    f"retrying in {wait:.0f}s"
                )
                self._sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError("unreachable")

    def _throttle(self) -> None:
        if self.min_interval_s <= 0:
            return
        elapsed = time.monotonic() - self._last_request
        if self._last_request and elapsed < self.min_interval_s:
            self._sleep(self.min_interval_s - elapsed)
        self._last_request = time.monotonic()

    @staticmethod
    def _to_hourly(rows: List[dict], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Hourly UTC mean of the raw points, keyed by period start, limited to [start, end)."""
        if not rows:
            return pd.Series(dtype=float)
        df = pd.DataFrame(rows)
        if "startTime" not in df or "value" not in df:
            return pd.Series(dtype=float)
        ts = pd.to_datetime(df["startTime"], utc=True, format="ISO8601")
        values = pd.to_numeric(df["value"], errors="coerce")
        s = pd.Series(values.values, index=ts).dropna()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        hourly = s.groupby(s.index.floor("h")).mean()
        hourly = hourly[(hourly.index >= start) & (hourly.index < end)]
        hourly.index.name = "timestamp"
        return hourly
