"""Train a forecast challenger against an existing history database.

The active model changes only if the challenger passes the recent point-error
and interval-quality checks. --spike opts into the experimental spike model.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.models  # noqa: F401
import app.forecasting.models  # noqa: F401
from app.database import Base
from app.forecasting import training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True)
    parser.add_argument("--model-dir", help="Defaults to models/ alongside the database")
    parser.add_argument("--spike", action="store_true")
    args = parser.parse_args()
    path = Path(args.db).resolve()
    if not path.is_file():
        parser.error("--db must point to an existing history database")
    training.MODEL_DIR = Path(args.model_dir).resolve() if args.model_dir else path.parent / "models"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        result = training.retrain(db, "manual-spike" if args.spike else "manual", spike_mixture=args.spike)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
