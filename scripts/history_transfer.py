"""Move backfilled forecasting history between databases without the APIs.

Copies only the raw history tables (spot_prices_raw, driver_data). Models and
predictions are not copied: the app retrains them on start (the walk-forward
bootstrap takes minutes once the history is present).

    # on the machine that ran the backfill
    PYTHONPATH=. python scripts/history_transfer.py export --db data/electricity_prices.db --out history.db

    # on the target, e.g. inside the container (data/ is the mounted volume)
    python scripts/history_transfer.py import history.db

Import merges into the target database and never overwrites rows that already
exist there, so it is safe to run against the live database and to repeat.
"""

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine  # noqa: E402

import app.models  # noqa: E402,F401
import app.forecasting.models  # noqa: E402,F401
from app.database import Base  # noqa: E402

TABLES = ("spot_prices_raw", "driver_data")
DEFAULT_DB = "data/electricity_prices.db"


def _create_tables(path: str) -> None:
    """Create the forecasting tables (and the rest of the app schema) if missing."""
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    engine.dispose()


def _copy(target: str, source: str, mode: str) -> dict:
    conn = sqlite3.connect(target)
    try:
        conn.execute("ATTACH DATABASE ? AS src", (source,))
        counts = {}
        with conn:
            for table in TABLES:
                cols = [row[1] for row in conn.execute(f"PRAGMA main.table_info({table})")]
                col_list = ", ".join(cols)
                before = conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
                conn.execute(
                    f"INSERT {mode} INTO main.{table} ({col_list}) SELECT {col_list} FROM src.{table}"
                )
                after = conn.execute(f"SELECT COUNT(*) FROM main.{table}").fetchone()[0]
                counts[table] = after - before
        conn.execute("DETACH DATABASE src")
        return counts
    finally:
        conn.close()


def export_history(db: str, out: str) -> dict:
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; remove it first")
    _create_tables(out)
    counts = _copy(out, db, "OR IGNORE")
    conn = sqlite3.connect(out)
    conn.execute("VACUUM")
    conn.close()
    return counts


def import_history(source: str, db: str) -> dict:
    _create_tables(db)
    return _copy(db, source, "OR IGNORE")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="write the history tables to a new file")
    exp.add_argument("--db", default=DEFAULT_DB, help=f"database to export from (default {DEFAULT_DB})")
    exp.add_argument("--out", required=True, help="new file to create")
    imp = sub.add_parser("import", help="merge an exported file into a database")
    imp.add_argument("file", help="file created by export")
    imp.add_argument("--db", default=DEFAULT_DB, help=f"database to import into (default {DEFAULT_DB})")
    args = parser.parse_args()

    if args.command == "export":
        counts = export_history(args.db, args.out)
        size_mb = os.path.getsize(args.out) / 1e6
        print(f"Exported {counts} to {args.out} ({size_mb:.1f} MB)")
    else:
        if not os.path.exists(args.file):
            raise SystemExit(f"{args.file} not found")
        counts = import_history(args.file, args.db)
        print(f"Imported new rows {counts} into {args.db}")


if __name__ == "__main__":
    main()
