import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.forecasting.models  # noqa: F401
import app.models  # noqa: F401
from app.database import Base
from app.forecasting import store

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "history_transfer.py"
HOURS = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")


def _session(path):
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _run(*args):
    out = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _count(path, table):
    with sqlite3.connect(path) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_export_then_import_merges_without_overwriting(tmp_path):
    source, target, export = tmp_path / "src.db", tmp_path / "live.db", tmp_path / "history.db"
    db = _session(source)
    store.upsert_spot_prices(db, pd.DataFrame({
        "timestamp": HOURS, "zone": "FI", "price_eur_mwh": [1.0, 2.0, 3.0], "resolution_minutes": 60,
    }))
    store.upsert_driver_data(db, pd.DataFrame({"timestamp": HOURS, "series": "wind_forecast_mw", "value": 5.0}), "x")
    db.close()

    live = _session(target)
    with sqlite3.connect(target) as conn:  # existing live data must survive
        conn.execute("INSERT INTO electricity_prices (timestamp, price) VALUES ('2025-01-01 00:00:00.000000', 9.9)")
    store.upsert_spot_prices(live, pd.DataFrame({
        "timestamp": HOURS[:1], "zone": "FI", "price_eur_mwh": [42.0], "resolution_minutes": 60,
    }))
    live.close()

    _run("export", "--db", str(source), "--out", str(export))
    assert "'spot_prices_raw': 2" in _run("import", str(export), "--db", str(target))
    assert "'spot_prices_raw': 0" in _run("import", str(export), "--db", str(target))  # idempotent

    assert _count(target, "electricity_prices") == 1
    assert _count(target, "spot_prices_raw") == 3
    assert _count(target, "driver_data") == 3
    with sqlite3.connect(target) as conn:
        kept = conn.execute("SELECT price_eur_mwh FROM spot_prices_raw ORDER BY timestamp LIMIT 1").fetchone()[0]
    assert kept == 42.0  # existing row not overwritten
