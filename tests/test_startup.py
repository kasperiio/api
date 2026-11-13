"""Exercise real app startup and an existing SQLite schema in a separate process."""

import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("existing_database", [False, True])
def test_real_startup_and_database_upgrade(tmp_path, existing_database):
    if existing_database:
        (tmp_path / "data").mkdir()
        with sqlite3.connect(tmp_path / "data/electricity_prices.db") as db:
            db.execute("CREATE TABLE electricity_prices (timestamp TIMESTAMP PRIMARY KEY NOT NULL, price FLOAT NOT NULL)")
            db.execute("INSERT INTO electricity_prices VALUES ('2026-01-01 00:00:00.000000', 4.2)")
    code = textwrap.dedent('''
        from fastapi.testclient import TestClient
        from sqlalchemy import text
        from main import app, engine
        with TestClient(app) as client:
            assert client.get('/docs').status_code == 200
            status = client.get('/forecast/status')
            assert status.status_code == 200, status.text
            assert status.json()['scheduler']['enabled'] is False
            assert client.get('/forecast/accuracy').status_code == 200
            assert client.get('/forecast').status_code == 404
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM archived_weather")).scalar() == 0
            assert conn.execute(text("SELECT COUNT(*) FROM outage_events")).scalar() == 0
            conn.execute(text("INSERT INTO electricity_prices (timestamp, price) VALUES ('2026-01-02', NULL)"))
        engine.dispose()
    ''')
    env = dict(os.environ, FORECAST_ENABLED="false", PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    if existing_database:
        with sqlite3.connect(tmp_path / "data/electricity_prices.db") as db:
            assert db.execute("SELECT price FROM electricity_prices ORDER BY timestamp LIMIT 1").fetchone()[0] == 4.2
