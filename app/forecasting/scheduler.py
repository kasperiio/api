"""
In-process scheduler that keeps the forecasting system learning.

Jobs (Europe/Helsinki time):
    hourly  :05        capture driver forecast snapshot
    daily   14:30      refresh recent data, score predictions, drift check (-> retrain)
    daily   14:45      issue + store the 7-day forecast
    weekly  Sun 03:00  champion/challenger retrain (+ re-issue forecast if promoted)
    startup +10 s      bootstrap if needed, refresh recent data, issue forecast

Disabled with FORECAST_ENABLED=false. Runs in a single uvicorn worker only
(multiple workers would each start a scheduler).
"""

import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from app import database
from app.forecasting import feedback, ingest, store, training
from app.logging_config import logger

TIMEZONE = "Europe/Helsinki"
DRIFT_RETRAIN_COOLDOWN = timedelta(days=3)

_scheduler: Optional[BackgroundScheduler] = None
_status_lock = threading.Lock()
_last_runs: Dict[str, dict] = {}


def is_enabled() -> bool:
    return os.getenv("FORECAST_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def _json_safe(value):
    """Keep job results small and serialisable for the status endpoint."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value][:50]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "__len__") and hasattr(value, "columns"):  # DataFrame
        return {"rows": len(value)}
    return str(value)


def _run(name: str, fn: Callable) -> None:
    """Run ``fn(db)`` with its own session, recording outcome and logging errors."""
    started = datetime.now(timezone.utc)
    with _status_lock:
        _last_runs.setdefault(name, {})["running"] = True
    db = database.SessionLocal()
    status, result, error = "ok", None, None
    try:
        result = fn(db)
    except Exception as e:
        status, error = "error", f"{type(e).__name__}: {e}"
        logger.exception("Forecast job %s failed", name)
        db.rollback()
    finally:
        db.close()
    with _status_lock:
        _last_runs[name] = {
            "running": False,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "result": _json_safe(result),
            "error": error,
        }


# --------------------------------------------------------------------------- job bodies

def _snapshot(db) -> dict:
    return {"rows": ingest.capture_snapshot(db)}


def _daily_refresh(db) -> dict:
    out = {}
    try:
        out["refresh"] = ingest.refresh_recent(db)
    except Exception as e:  # still score/check drift with what we have
        logger.exception("Daily refresh failed")
        db.rollback()
        out["refresh_error"] = str(e)
    out["score"] = feedback.score(db)
    drift = feedback.drift_status(db)
    out["drift"] = drift
    if drift["drift"]:
        last = training.latest_model_created(db)
        if last is not None and datetime.now(timezone.utc) - last < DRIFT_RETRAIN_COOLDOWN:
            logger.warning("Drift detected but a model was trained at %s; cooling down", last)
            out["retrain"] = "cooldown"
        else:
            logger.warning("Drift detected (%s); retraining", drift)
            out["retrain"] = training.retrain(db, "drift", spike_mixture=True)
            if out["retrain"].get("activated"):
                training.predict_and_store(db)
    return out


def _predict(db) -> dict:
    if store.get_active_model(db) is None:
        return {"skipped": "no active model"}
    out = {}
    # Future driver values only exist in snapshots; take a fresh one so the
    # forecast never depends on the hourly job having run (e.g. right after startup).
    try:
        out["snapshot_rows"] = ingest.capture_snapshot(db)
    except Exception as e:
        logger.exception("Snapshot before prediction failed")
        db.rollback()
        out["snapshot_error"] = str(e)
    out["hours"] = len(training.predict_and_store(db))
    return out


def _weekly_retrain(db) -> dict:
    result = training.retrain(db, "weekly", spike_mixture=True)
    if result.get("activated"):
        training.predict_and_store(db)
    return result


def _startup(db) -> dict:
    out = {"bootstrap": training.ensure_bootstrapped(db)}
    if out["bootstrap"].get("status") == "already_bootstrapped":
        try:
            out["refresh"] = ingest.refresh_recent(db)
            out["score"] = feedback.score(db)
        except Exception as e:  # a failed refresh must not block the forecast
            logger.exception("Startup refresh failed")
            db.rollback()
            out["refresh_error"] = str(e)
    out["predict"] = _predict(db)
    return out


JOBS = {
    "snapshot": (_snapshot, CronTrigger(minute=5, timezone=TIMEZONE), 30 * 60),
    "daily_refresh": (_daily_refresh, CronTrigger(hour=14, minute=30, timezone=TIMEZONE), 6 * 3600),
    "daily_predict": (_predict, CronTrigger(hour=14, minute=45, timezone=TIMEZONE), 6 * 3600),
    "weekly_retrain": (_weekly_retrain,
                       CronTrigger(day_of_week="sun", hour=3, minute=0, timezone=TIMEZONE), 12 * 3600),
}


def start_scheduler() -> Optional[BackgroundScheduler]:
    """Start the background scheduler (no-op when disabled or already running)."""
    global _scheduler
    if not is_enabled():
        logger.info("Forecast scheduler disabled (FORECAST_ENABLED=false)")
        return None
    if _scheduler is not None and _scheduler.running:
        return _scheduler
    sched = BackgroundScheduler(timezone=TIMEZONE)
    for name, (fn, trigger, grace) in JOBS.items():
        sched.add_job(
            _run, trigger, args=[name, fn], id=name, name=name,
            max_instances=1, coalesce=True, misfire_grace_time=grace, replace_existing=True,
        )
    sched.add_job(
        _run, DateTrigger(run_date=datetime.now(timezone.utc) + timedelta(seconds=10)),
        args=["startup", _startup], id="startup", name="startup",
        max_instances=1, coalesce=True, misfire_grace_time=3600,
    )
    sched.start()
    _scheduler = sched
    logger.info("Forecast scheduler started")
    return sched


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        try:
            _scheduler.shutdown(wait=False)
        finally:
            _scheduler = None
            logger.info("Forecast scheduler stopped")


def get_status() -> dict:
    jobs = []
    if _scheduler is not None:
        for job in _scheduler.get_jobs():
            nrt = job.next_run_time
            jobs.append({"id": job.id, "next_run_time": nrt.isoformat() if nrt else None})
    with _status_lock:
        last = {k: dict(v) for k, v in _last_runs.items()}
    return {
        "enabled": is_enabled(),
        "running": bool(_scheduler is not None and _scheduler.running),
        "jobs": jobs,
        "last_runs": last,
    }
