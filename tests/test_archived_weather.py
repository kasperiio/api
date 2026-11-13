import numpy as np
import pandas as pd

from app.forecasting import features as F, series as S, store
from test_features import synthetic_data
from test_store import db


def test_archive_roundtrip_and_upsert(db):
    t = pd.Timestamp("2025-03-01", tz="UTC")
    frame = pd.DataFrame({"timestamp": [t], "series": [S.TEMPERATURE_FORECAST_C],
                          "lead_days": [2], "value": [7.0]})
    store.upsert_archived_weather(db, frame)
    frame["value"] = 8.0
    store.upsert_archived_weather(db, frame)
    got = store.load_archived_weather(db, t, t + pd.Timedelta(hours=1))
    assert len(got) == 1
    assert got.iloc[0].value == 8.0


def test_archive_uses_older_runs_and_snapshot_takes_precedence():
    prices, drivers, _ = synthetic_data(days=100)
    drivers = drivers.reindex(columns=F.ALL_SERIES)
    issue = pd.Timestamp("2025-03-01 12:00", tz="UTC")
    target = issue + pd.Timedelta(hours=24)
    archive = pd.DataFrame([
        dict(timestamp=target, series=S.TEMPERATURE_FORECAST_C, lead_days=1, value=99.0),
        dict(timestamp=target, series=S.TEMPERATURE_FORECAST_C, lead_days=2, value=-12.0),
    ])
    kwargs = dict(prices=prices, drivers=drivers, issues=pd.DatetimeIndex([issue]),
                  noise=True, seed=42, archived_weather=archive)
    frame = F._assemble(**kwargs)
    assert frame.iloc[23][S.TEMPERATURE_FORECAST_C] == -12.0
    # Archive replacement happens before derived daily features.
    assert np.isfinite(frame.iloc[23]["temperature_day_mean"])
    snap = pd.DataFrame({S.TEMPERATURE_FORECAST_C: [3.0]}, index=pd.DatetimeIndex([target]))
    frame = F._assemble(**kwargs, snapshots={0: snap})
    assert frame.iloc[23][S.TEMPERATURE_FORECAST_C] == 3.0
