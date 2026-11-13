"""Data ingestion: historical backfill, daily refresh and hourly forecast snapshots.

Sources are looked up through :func:`get_sources`, which returns a dict of
instantiated, available clients keyed by ``"entsoe"``, ``"fingrid"`` and
``"openmeteo"``. Every public function also accepts ``sources=`` so tests (or
callers) can inject their own objects with the same method signatures.

All writes go through ``app.forecasting.store``; every write is an upsert, so
running any of these functions twice is harmless.
"""

import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting import series as S
from app.forecasting import store
from app.logging_config import logger

SNAPSHOT_HORIZON = timedelta(days=8)
COVERAGE_THRESHOLD = 0.95

Sources = Dict[str, object]


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def _build_fingrid():
    from app.forecasting.sources.fingrid import FingridClient
    return FingridClient()


def _build_entsoe():
    from app.forecasting.sources.entsoe_data import EntsoeDataClient
    return EntsoeDataClient()


def _build_openmeteo():
    from app.forecasting.sources.openmeteo import OpenMeteoClient
    return OpenMeteoClient()


def _build_commodities():
    from app.forecasting.sources.commodities import CommoditiesClient
    return CommoditiesClient()


_BUILDERS: Dict[str, Callable[[], object]] = {
    "entsoe": _build_entsoe,
    "fingrid": _build_fingrid,
    "openmeteo": _build_openmeteo,
    "commodities": _build_commodities,
}


def get_sources() -> Sources:
    """Instantiate all source clients and return the ones that are usable.

    Sources whose ``is_available()`` is False (e.g. missing API key) or whose
    construction fails are skipped with a warning.
    """
    available: Sources = {}
    for name, build in _BUILDERS.items():
        try:
            client = build()
            is_available = getattr(client, "is_available", None)
            if is_available is not None and not is_available():
                logger.warning("Forecast ingest: source %s unavailable (missing API key?), skipping", name)
                continue
            available[name] = client
        except Exception as exc:  # noqa: BLE001 - a broken source must not stop the others
            logger.warning("Forecast ingest: could not initialise source %s: %s", name, exc)
    return available


def _filter_available(sources: Sources) -> Sources:
    """Drop injected sources that report themselves unavailable."""
    out: Sources = {}
    for name, client in sources.items():
        is_available = getattr(client, "is_available", None)
        try:
            ok = is_available() if is_available is not None else True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Forecast ingest: availability check failed for %s: %s", name, exc)
            ok = False
        if ok:
            out[name] = client
        else:
            logger.warning("Forecast ingest: source %s unavailable, skipping", name)
    return out


def _resolve(sources: Optional[Sources]) -> Sources:
    return get_sources() if sources is None else _filter_available(sources)


# ---------------------------------------------------------------------------
# Jobs: one (source, kind) pair that fetches a frame and upserts it
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Job:
    name: str            # key in the result counts
    source: str          # key in the sources dict
    kind: str            # "prices" or "drivers"
    method: str          # client method used for historical data
    coverage_keys: Tuple[str, ...]  # zones / series that must be >=95% present
    chunk_months: int = 1  # backfill window size


# Coverage keys are the series each source reliably provides for every hour;
# sparse series (weekly reservoirs, capacity) are deliberately not included.
# Fingrid is limited to 10 requests/minute but pages 20k rows per request, so
# it gets large windows: fewer requests per dataset.
_JOBS: Tuple[_Job, ...] = (
    _Job("entsoe_prices", "entsoe", "prices", "fetch_prices", tuple(S.ZONES)),
    _Job("entsoe_drivers", "entsoe", "drivers", "fetch_drivers", (S.LOAD_FORECAST_MW,)),
    _Job("fingrid", "fingrid", "drivers", "fetch_drivers",
         (S.CONSUMPTION_ACTUAL_MW, S.NUCLEAR_ACTUAL_MW), chunk_months=6),  # 15-min and 3-min data
    _Job("openmeteo", "openmeteo", "drivers", "fetch_historical_forecasts", (S.TEMPERATURE_FORECAST_C,)),
)


def _to_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Split [start, end) at UTC month boundaries."""
    chunks = []
    cur = start
    while cur < end:
        nxt = (cur.normalize().replace(day=1) + pd.DateOffset(months=1))
        nxt = min(nxt, end)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks


def _merge_windows(
    months: List[Tuple[pd.Timestamp, pd.Timestamp]], max_months: int
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Merge consecutive month chunks into windows of at most ``max_months``."""
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    run = 0
    for c_start, c_end in months:
        if windows and run < max_months and windows[-1][1] == c_start:
            windows[-1] = (windows[-1][0], c_end)
            run += 1
        else:
            windows.append((c_start, c_end))
            run = 1
    return windows


def _expected_hours(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return len(pd.date_range(start.ceil("h"), end, freq="h", inclusive="left"))


def _is_covered(db: Session, job: _Job, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    expected = _expected_hours(start, end)
    if expected == 0:
        return True
    if job.kind == "prices":
        wide = store.load_spot_prices(db, start, end, zones=job.coverage_keys)
    else:
        wide = store.load_drivers(db, start, end, series=list(job.coverage_keys))
    for key in job.coverage_keys:
        if key not in wide.columns or wide[key].notna().sum() < COVERAGE_THRESHOLD * expected:
            return False
    return True


def _store(db: Session, job: _Job, df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty:
        return 0
    if job.kind == "prices":
        return store.upsert_spot_prices(db, df)
    return store.upsert_driver_data(db, df.dropna(subset=["value"]), source=job.source)


def _run_job(db: Session, job: _Job, client, start, end, method: Optional[str] = None) -> int:
    fetch = getattr(client, method or job.method)
    df = fetch(start.to_pydatetime(), end.to_pydatetime())
    return _store(db, job, df)


def _empty_result(jobs) -> dict:
    return {"counts": {j.name: 0 for j in jobs}, "skipped_chunks": {j.name: 0 for j in jobs},
            "failures": [], "unavailable": []}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def backfill(db: Session, start: datetime, end: datetime, sources: Optional[Sources] = None) -> dict:
    """Fill spot_prices_raw and driver_data for [start, end) from all available sources.

    Coverage is checked per UTC month: a (month, job) pair is skipped when its
    coverage series already have >=95% of the expected hours, so an interrupted
    run can simply be restarted. Consecutive missing months are fetched together
    in windows of up to ``_Job.chunk_months``. Jobs fetch in parallel,
    one thread each (each source keeps its own rate limit); all writes happen on
    the calling thread because SQLite has a single writer. A failing window is
    logged and recorded in ``failures``; the run continues.

    Returns ``{"counts": {job: rows}, "skipped_chunks": {job: n},
    "failures": [{"job", "start", "end", "error"}], "unavailable": [source]}``.
    """
    srcs = _resolve(sources)
    start_ts, end_ts = _to_utc(start), _to_utc(end)
    result = _empty_result(_JOBS)
    jobs = []
    for job in _JOBS:
        if job.source in srcs:
            jobs.append(job)
        elif job.source not in result["unavailable"]:
            result["unavailable"].append(job.source)
    if result["unavailable"]:
        logger.warning("Backfill: sources not available, skipped: %s", ", ".join(result["unavailable"]))

    def failure(job, c_start, c_end, exc):
        result["failures"].append({
            "job": job.name, "start": c_start.isoformat(), "end": c_end.isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        })

    # Coverage checks read the DB, so they run here before any thread starts.
    pending: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    months = _month_chunks(start_ts, end_ts)
    for job in jobs:
        missing = []
        for c_start, c_end in months:
            try:
                covered = _is_covered(db, job, c_start, c_end)
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                logger.error("Backfill %s coverage check failed: %s", job.name, exc)
                covered = False
            if covered:
                result["skipped_chunks"][job.name] += 1
            else:
                missing.append((c_start, c_end))
        pending[job.name] = _merge_windows(missing, job.chunk_months)
    logger.info("Backfill %s .. %s: windows to fetch %s", start_ts.isoformat(), end_ts.isoformat(),
                {name: len(w) for name, w in pending.items()})

    fetched: "queue.Queue" = queue.Queue()
    _DONE = object()

    def worker(job: _Job) -> None:
        fetch = getattr(srcs[job.source], job.method)
        for c_start, c_end in pending[job.name]:
            try:
                fetched.put((job, c_start, c_end, fetch(c_start.to_pydatetime(), c_end.to_pydatetime()), None))
            except Exception as exc:  # noqa: BLE001
                fetched.put((job, c_start, c_end, None, exc))
        fetched.put(_DONE)

    threads = [threading.Thread(target=worker, args=(job,), daemon=True, name=f"backfill-{job.name}")
               for job in jobs if pending[job.name]]
    for t in threads:
        t.start()
    remaining = len(threads)
    done_windows: Dict[str, int] = {name: 0 for name in pending}
    while remaining:
        item = fetched.get()
        if item is _DONE:
            remaining -= 1
            continue
        job, c_start, c_end, df, exc = item
        done_windows[job.name] += 1
        label = (f"[{done_windows[job.name]}/{len(pending[job.name])}] "
                 f"{job.name} {c_start:%Y-%m-%d}..{c_end:%Y-%m-%d}")
        if exc is None:
            try:
                n = _store(db, job, df)
                result["counts"][job.name] += n
                logger.info("Backfill %s: %d rows", label, n)
                continue
            except Exception as store_exc:  # noqa: BLE001
                db.rollback()
                exc = store_exc
        logger.error("Backfill %s failed: %s", label, exc)
        failure(job, c_start, c_end, exc)
    for t in threads:
        t.join()
    logger.info("Backfill done: counts=%s failures=%d", result["counts"], len(result["failures"]))
    result["additional"] = refresh_additional(db, start_ts, end_ts, sources=srcs)
    return result


def refresh_recent(
    db: Session,
    days_back: int = 3,
    days_ahead: int = 2,
    sources: Optional[Sources] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Update prices and driver data for [now - days_back, now + days_ahead).

    No coverage check: recent hours are always refetched so revised actuals and
    the latest forecasts overwrite older values. Open-Meteo uses the archived
    forecast API for past hours and the live forecast API for future hours.
    """
    srcs = _resolve(sources)
    now_ts = _to_utc(now or datetime.now(timezone.utc)).floor("h")
    start = now_ts - pd.Timedelta(days=days_back)
    end = now_ts + pd.Timedelta(days=days_ahead)
    result = _empty_result(_JOBS)

    for job in _JOBS:
        if job.source not in srcs:
            if job.source not in result["unavailable"]:
                result["unavailable"].append(job.source)
            continue
        client = srcs[job.source]
        if job.source == "openmeteo":
            parts = [("fetch_historical_forecasts", start, now_ts), ("fetch_forecasts", now_ts, end)]
        else:
            parts = [(job.method, start, end)]
        for method, p_start, p_end in parts:
            if p_start >= p_end:
                continue
            try:
                n = _run_job(db, job, client, p_start, p_end, method=method)
                result["counts"][job.name] += n
            except Exception as exc:  # noqa: BLE001
                db.rollback()
                logger.error("Refresh %s (%s) failed: %s", job.name, method, exc)
                result["failures"].append({
                    "job": job.name, "start": p_start.isoformat(), "end": p_end.isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                })
    logger.info("Refresh recent done: counts=%s failures=%d", result["counts"], len(result["failures"]))
    result["additional"] = refresh_additional(db, start, end, sources=srcs)
    return result


def refresh_additional(db: Session, start: datetime, end: datetime,
                       sources: Optional[Sources] = None) -> dict:
    """Persist forecast-age archives and event revisions without flattening them.

    These sparse sources have no hourly coverage test: re-fetching a window is
    intentional, since outage revisions and archive gaps can arrive later.
    """
    srcs = _resolve(sources)
    result = {"counts": {}, "failures": []}
    for name, method, save in (
        ("openmeteo", "fetch_previous_runs", store.upsert_archived_weather),
        ("entsoe", "fetch_outages", store.upsert_outages),
        ("entsoe", "fetch_transfer_capacities", lambda db, frame: store.upsert_driver_data(db, frame, "entsoe")),
        ("commodities", "fetch_drivers", lambda db, frame: store.upsert_driver_data(db, frame, "eex")),
    ):
        fetch = getattr(srcs.get(name), method, None)
        if fetch is None:
            continue
        lo, hi = _to_utc(start), _to_utc(end)
        result["counts"][method] = 0
        while lo < hi:
            # Commit each archive window independently so a later HTTP failure
            # cannot discard hours of successful historical source work.
            nxt = min(lo + pd.DateOffset(months=3), hi) if name != "commodities" else hi
            try:
                frame = fetch(lo.to_pydatetime(), nxt.to_pydatetime())
                count = save(db, frame) if frame is not None and not frame.empty else 0
                result["counts"][method] += count
            except Exception as exc:
                db.rollback()
                logger.error("Additional ingest %s %s..%s failed: %s", method, lo, nxt, exc)
                result["failures"].append({"job": method, "start": lo.isoformat(),
                                           "end": nxt.isoformat(), "error": f"{type(exc).__name__}: {exc}"})
            lo = nxt
    return result


def capture_snapshot(
    db: Session, issued_at: datetime = None, sources: Optional[Sources] = None
) -> int:
    """Fetch current forecasts for issued_at..issued_at+8 days from all sources and store
    them as a forecast snapshot at ``issued_at`` (default now, floored to the hour).

    The same values are also upserted into ``driver_data`` so it always holds the
    latest forecast. Returns the number of snapshot rows stored.
    """
    srcs = _resolve(sources)
    issued = _to_utc(issued_at or datetime.now(timezone.utc)).floor("h")
    end = issued + pd.Timedelta(SNAPSHOT_HORIZON)

    frames = []
    for name, client in srcs.items():
        capacity_fetch = getattr(client, "fetch_transfer_capacities", None)
        if capacity_fetch is not None:
            try:
                capacity = capacity_fetch(issued.to_pydatetime(), end.to_pydatetime())
                if capacity is not None and not capacity.empty:
                    frames.append(capacity)
            except Exception as exc:
                logger.error("Snapshot: %s transfer capacity refresh failed: %s", name, exc)
        outage_fetch = getattr(client, "fetch_outages", None)
        if outage_fetch is not None:
            try:
                events = outage_fetch(issued.to_pydatetime(), end.to_pydatetime())
                if events is not None and not events.empty:
                    store.upsert_outages(db, events)
            except Exception as exc:
                db.rollback()
                logger.error("Snapshot: %s outage refresh failed: %s", name, exc)
        fetch = getattr(client, "fetch_forecasts", None)
        if fetch is None:
            continue
        try:
            df = fetch(issued.to_pydatetime(), end.to_pydatetime())
        except Exception as exc:  # noqa: BLE001
            logger.error("Snapshot: %s fetch_forecasts failed: %s", name, exc)
            continue
        if df is None or df.empty:
            continue
        df = df[["timestamp", "series", "value"]].dropna(subset=["value"])
        try:
            store.upsert_driver_data(db, df, source=name)
        except Exception as exc:  # noqa: BLE001
            db.rollback()
            logger.error("Snapshot: storing %s driver data failed: %s", name, exc)
        frames.append(df)

    if not frames:
        logger.warning("Snapshot at %s: no forecast data from any source", issued.isoformat())
        return 0
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.drop_duplicates(["timestamp", "series"], keep="last")
    n = store.insert_snapshot(db, combined, issued.to_pydatetime())
    logger.info("Snapshot at %s: %d rows from %s", issued.isoformat(), n, list(srcs))
    return n
