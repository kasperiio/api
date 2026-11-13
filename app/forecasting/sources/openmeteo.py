"""Open-Meteo weather client (no API key).

Two endpoints, same parameters and response shape:

- Historical Forecast API (``historical-forecast-api.open-meteo.com/v1/forecast``):
  archived model forecasts from ~2022 onwards, queried with start_date/end_date.
- Forecast API (``api.open-meteo.com/v1/forecast``): current forecast, up to 16
  days ahead (``forecast_days``) plus up to 92 days back (``past_days``).

Multiple locations are requested in one call via comma-separated
latitude/longitude; the response is then a JSON list (one object per location,
in request order) instead of a single object. ``timezone=UTC`` makes ``time``
naive ISO strings in UTC.

Produced series (hourly, UTC):

- ``temperature_forecast_c``: population-weighted mean of ``temperature_2m`` over
  six cities (weights = approx. municipal populations, normalised):
  Helsinki capital region (Helsinki+Espoo+Vantaa) 1250k, Tampere 255k,
  Oulu 215k, Turku 205k, Jyväskylä 148k, Kuopio 125k.
- ``wind_speed_100m_forecast_ms``: plain mean of ``wind_speed_100m`` (m/s) over
  five points in the main wind-farm belt along the Bothnian coast
  (Kristinestad, Pori, Kalajoki, Raahe, Simo).
- ``solar_radiation_forecast_wm2``: plain mean of ``shortwave_radiation`` (W/m2)
  over the six city points. Note Open-Meteo radiation is the mean over the
  *preceding* hour (value at 12:00 covers 11:00-12:00).
"""

import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

from app.forecasting import series as S
from app.logging_config import logger

# (name, lat, lon, weight)
CITIES: List[Tuple[str, float, float, float]] = [
    ("Helsinki", 60.17, 24.94, 1250.0),
    ("Tampere", 61.50, 23.76, 255.0),
    ("Oulu", 65.01, 25.47, 215.0),
    ("Turku", 60.45, 22.27, 205.0),
    ("Jyvaskyla", 62.24, 25.75, 148.0),
    ("Kuopio", 62.89, 27.68, 125.0),
]

WIND_POINTS: List[Tuple[str, float, float]] = [
    ("Kristinestad", 62.27, 21.38),
    ("Pori", 61.48, 21.80),
    ("Kalajoki", 64.26, 23.95),
    ("Raahe", 64.68, 24.48),
    ("Simo", 65.66, 25.06),
]

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


class OpenMeteoClient:
    HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_START = pd.Timestamp("2022-01-01", tz="UTC")
    CHUNK_MONTHS = 3
    MAX_FORECAST_DAYS = 16
    MAX_PAST_DAYS = 92
    MAX_RETRIES = 3
    TIMEOUT = (10, 120)

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        sleep: Callable[[float], None] = time.sleep,
        backoff_s: float = 5.0,
    ):
        self.session = session or requests.Session()
        self._sleep = sleep
        self.backoff_s = backoff_s

    def fetch_historical_forecasts(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Driver frame of archived weather forecasts (historical-forecast API) for [start, end)."""
        start_ts, end_ts = _to_utc(start).floor("h"), _to_utc(end)
        if start_ts < self.HISTORICAL_START:
            logger.info("Open-Meteo: historical forecasts start 2022-01-01, clipping request")
            start_ts = self.HISTORICAL_START
        if end_ts <= start_ts:
            return _empty()

        frames = []
        chunk_start = start_ts
        while chunk_start < end_ts:
            chunk_end = min(chunk_start + pd.DateOffset(months=self.CHUNK_MONTHS), end_ts)
            params = {
                "start_date": chunk_start.strftime("%Y-%m-%d"),
                # end_date is inclusive
                "end_date": (chunk_end - pd.Timedelta(seconds=1)).strftime("%Y-%m-%d"),
            }
            try:
                frames.append(self._fetch_all(self.HISTORICAL_URL, params, chunk_start, chunk_end))
            except Exception as e:
                logger.warning(
                    f"Open-Meteo: historical chunk {params['start_date']}..{params['end_date']} "
                    f"failed, skipping: {e}"
                )
            chunk_start = chunk_end
        return self._concat(frames)

    def fetch_forecasts(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Driver frame of the current weather forecast for [start, end)."""
        start_ts, end_ts = _to_utc(start).floor("h"), _to_utc(end)
        if end_ts <= start_ts:
            return _empty()
        today = pd.Timestamp.now(tz="UTC").floor("D")
        past_days = int(max(0, min(self.MAX_PAST_DAYS, (today - start_ts.floor("D")).days)))
        params = {"forecast_days": self.MAX_FORECAST_DAYS, "past_days": past_days}
        try:
            return self._concat([self._fetch_all(self.FORECAST_URL, params, start_ts, end_ts)])
        except Exception as e:
            logger.warning(f"Open-Meteo: forecast fetch failed: {e}")
            return _empty()

    # ------------------------------------------------------------------ internals

    def _fetch_all(
        self, url: str, base_params: dict, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Fetch city + wind-point groups and build the three aggregate series."""
        cities = self._fetch_points(
            url, base_params, [(c[1], c[2]) for c in CITIES],
            ["temperature_2m", "shortwave_radiation"],
        )
        wind = self._fetch_points(
            url, base_params, [(p[1], p[2]) for p in WIND_POINTS], ["wind_speed_100m"],
        )
        weights = pd.Series([c[3] for c in CITIES], dtype=float)

        out: Dict[str, pd.Series] = {
            S.TEMPERATURE_FORECAST_C: self._weighted_mean(
                [f.get("temperature_2m") for f in cities], weights
            ),
            S.SOLAR_RADIATION_FORECAST_WM2: self._weighted_mean(
                [f.get("shortwave_radiation") for f in cities], None
            ),
            S.WIND_SPEED_100M_FORECAST_MS: self._weighted_mean(
                [f.get("wind_speed_100m") for f in wind], None
            ),
        }
        frames = []
        for name, s in out.items():
            s = s[(s.index >= start) & (s.index < end)].dropna()
            if s.empty:
                continue
            frames.append(pd.DataFrame({
                "timestamp": s.index, "series": name, "value": s.values.astype(float)
            }))
        return self._concat(frames)

    @staticmethod
    def _weighted_mean(columns: List[Optional[pd.Series]], weights: Optional[pd.Series]) -> pd.Series:
        """Row-wise (weighted) mean across locations, ignoring missing locations/values."""
        present = [c.rename(i) for i, c in enumerate(columns) if c is not None]
        if not present:
            return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
        data = pd.concat(present, axis=1)
        if weights is None:
            return data.mean(axis=1, skipna=True)
        w = weights.reindex(data.columns).fillna(0.0).values
        total_w = pd.Series((data.notna().values * w).sum(axis=1), index=data.index)
        weighted = pd.Series((data.fillna(0.0).values * w).sum(axis=1), index=data.index)
        return weighted / total_w.where(total_w > 0)

    def _fetch_points(
        self,
        url: str,
        base_params: dict,
        points: List[Tuple[float, float]],
        variables: List[str],
    ) -> List[Dict[str, pd.Series]]:
        """One request for several coordinates; returns per-location {variable: series}."""
        params = dict(base_params)
        params.update({
            "latitude": ",".join(f"{lat:.4f}" for lat, _ in points),
            "longitude": ",".join(f"{lon:.4f}" for _, lon in points),
            "hourly": ",".join(variables),
            "wind_speed_unit": "ms",
            "timezone": "UTC",
        })
        payload = self._get(url, params)
        locations = payload if isinstance(payload, list) else [payload]
        result = []
        for loc in locations:
            hourly = loc.get("hourly") or {}
            times = hourly.get("time") or []
            index = pd.to_datetime(times, utc=True)
            per_var = {}
            for var in variables:
                vals = hourly.get(var)
                if vals is None:
                    continue
                per_var[var] = pd.Series(
                    pd.to_numeric(pd.Series(vals), errors="coerce").values, index=index
                )
            result.append(per_var)
        return result

    def _get(self, url: str, params: dict):
        for attempt in range(self.MAX_RETRIES + 1):
            resp = self.session.get(url, params=params, timeout=self.TIMEOUT)
            if (resp.status_code == 429 or resp.status_code >= 500) and attempt < self.MAX_RETRIES:
                wait = self.backoff_s * (2 ** attempt)
                logger.info(f"Open-Meteo: HTTP {resp.status_code}, retrying in {wait:.0f}s")
                self._sleep(wait)
                continue
            if resp.status_code >= 400:
                try:
                    reason = resp.json().get("reason")
                except Exception:
                    reason = None
                raise requests.HTTPError(
                    f"Open-Meteo HTTP {resp.status_code}: {reason or resp.text[:200]}"
                )
            return resp.json()
        raise RuntimeError("unreachable")

    @staticmethod
    def _concat(frames: List[pd.DataFrame]) -> pd.DataFrame:
        frames = [f for f in frames if f is not None and not f.empty]
        if not frames:
            return _empty()
        df = pd.concat(frames, ignore_index=True)[COLUMNS]
        return df.drop_duplicates(["timestamp", "series"], keep="last").reset_index(drop=True)
