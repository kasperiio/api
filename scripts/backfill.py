"""Backfill forecasting tables (spot prices + driver data) from all available sources.

Usage:
    PYTHONPATH=. venv/Scripts/python scripts/backfill.py --start 2023-01-01 [--end 2026-11-01]

Resumable: monthly chunks that are already stored are skipped, so the script can
be re-run after an interruption.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv


def _parse_date(value: str) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC").to_pydatetime()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", required=True, help="Start date (UTC), e.g. 2023-01-01")
    parser.add_argument("--end", default=None, help="End date (UTC, exclusive); default now")
    args = parser.parse_args(argv)

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import app.models  # noqa: F401  - register tables
    import app.forecasting.models  # noqa: F401
    from app.database import Base, SessionLocal, engine
    from app.forecasting.ingest import backfill

    Base.metadata.create_all(bind=engine)

    start = _parse_date(args.start)
    end = _parse_date(args.end) if args.end else datetime.now(timezone.utc)

    db = SessionLocal()
    try:
        result = backfill(db, start, end)
    finally:
        db.close()

    print(json.dumps(result, indent=2))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
