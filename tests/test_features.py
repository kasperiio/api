"""Tests for the forecasting feature pipeline (synthetic SQLite DB)."""

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.models  # noqa: F401  (registers TZDateTime tables on Base)
import app.forecasting.models  # noqa: F401
from app.database import Base
from app.forecasting import features as F
from app.forecasting import series as S
from app.forecasting.models import DriverData, ForecastSnapshot, SpotPriceRaw

SYN_START = pd.Timestamp("2025-01-01", tz="UTC")


# ---------------------------------------------------------------------------
# Synthetic data (shared with test_model / test_backtest)
# ---------------------------------------------------------------------------

def synthetic_data(days: int = 200, seed: int = 0):
    """Hourly synthetic drivers and prices whose level is driven by residual load."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(SYN_START, periods=days * 24, freq="h", tz="UTC")
    n = len(idx)
    hel = idx.tz_convert("Europe/Helsinki")
    hour = hel.hour.values
    doy = hel.dayofyear.values
    t = np.arange(n)

    temp = -5 + 12 * np.sin(2 * np.pi * (doy - 110) / 365) + 3 * np.sin(2 * np.pi * (hour - 9) / 24)
    temp = temp + np.cumsum(rng.normal(0, 0.3, n)) * 0.2
    weekend = hel.dayofweek.values >= 5
    cons = 9000 - 120 * temp + 900 * np.sin(np.pi * np.clip(hour - 6, 0, 16) / 16) - 700 * weekend
    cons = cons + rng.normal(0, 100, n)
    cf = np.empty(n)
    x = 0.0
    for i in range(n):  # mean-reverting wind capacity factor
        x = 0.985 * x + rng.normal(0, 0.12)
        cf[i] = 1 / (1 + np.exp(-(x - 0.6)))
    capacity = 8000 + 2 * (t // 168) * 10.0
    wind = capacity * cf * 0.85
    solar = np.clip(np.sin(np.pi * (hour - 5) / 14), 0, None) * 600 * np.clip((doy - 60) / 120, 0, 1)
    resid = cons - wind - solar
    nuclear = np.where((t // 24) % 60 < 8, 2800, 4300) + rng.normal(0, 30, n)
    eff_resid = resid - (nuclear - 4300)
    price = 10 * np.sinh((eff_resid - 4500) / 1800) + 25 + rng.normal(0, 5, n)
    price = np.clip(price, -500, 4000)

    drivers = pd.DataFrame({
        S.WIND_FORECAST_MW: wind,
        S.WIND_ACTUAL_MW: wind * (1 + rng.normal(0, 0.03, n)),
        S.WIND_CAPACITY_MW: capacity,
        S.SOLAR_FORECAST_MW: solar,
        S.CONSUMPTION_FORECAST_MW: cons,
        S.CONSUMPTION_ACTUAL_MW: cons + rng.normal(0, 50, n),
        S.TEMPERATURE_FORECAST_C: temp,
        S.WIND_SPEED_100M_FORECAST_MS: 3 + 9 * cf,
        S.NUCLEAR_ACTUAL_MW: nuclear,
        S.NET_IMPORT_ACTUAL_MW: np.clip(resid - 4000, -2000, 3000),
        S.LOAD_FORECAST_MW: cons * 1.01,
    }, index=idx)
    # Weekly reservoir, stamped at week start
    weekly = pd.date_range(SYN_START, idx[-1], freq="7D", tz="UTC")
    res = pd.DataFrame({
        S.HYDRO_RESERVOIR_NO_MWH: 50e6 + 10e6 * np.sin(np.arange(len(weekly)) / 8),
    }, index=weekly)
    prices = pd.DataFrame({
        "FI": price,
        "SE1": 0.8 * price + rng.normal(0, 3, n),
        "SE3": 0.9 * price + 10 + rng.normal(0, 5, n),
        "EE": price + 15 + rng.normal(0, 8, n),
    }, index=idx)
    return prices, drivers, res


def make_session(tmp_path, days: int = 200, seed: int = 0, prices=None):
    engine = create_engine("sqlite:///%s" % (tmp_path / "syn.db"),
                           connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    p, d, res = synthetic_data(days, seed)
    if prices is not None:
        p = prices
    write_prices(db, p)
    now = datetime.now(timezone.utc)
    rows = []
    for frame in (d, res):
        for col in frame.columns:
            s = frame[col].dropna()
            rows.extend({"timestamp": ts.to_pydatetime(), "series": col, "value": float(v),
                         "source": "synthetic", "updated_at": now} for ts, v in s.items())
    db.execute(DriverData.__table__.insert(), rows)
    db.commit()
    return db


def write_prices(db, prices: pd.DataFrame):
    db.execute(SpotPriceRaw.__table__.delete())
    rows = []
    for zone in prices.columns:
        rows.extend({"timestamp": ts.to_pydatetime(), "zone": zone, "price_eur_mwh": float(v),
                     "resolution_minutes": 60} for ts, v in prices[zone].dropna().items())
    db.execute(SpotPriceRaw.__table__.insert(), rows)
    db.commit()


@pytest.fixture(scope="module")
def syn_db(tmp_path_factory):
    db = make_session(tmp_path_factory.mktemp("feat"), days=60)
    yield db
    db.close()


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

def test_easter_dates():
    assert F.easter_sunday(2024) == date(2024, 3, 31)
    assert F.easter_sunday(2025) == date(2025, 4, 20)
    assert F.easter_sunday(2026) == date(2026, 4, 5)


def test_finnish_holidays_2025():
    h = F.finnish_holidays(2025)
    expected = {
        date(2025, 1, 1), date(2025, 1, 6), date(2025, 4, 18), date(2025, 4, 20),
        date(2025, 4, 21), date(2025, 5, 1), date(2025, 5, 29), date(2025, 6, 8),
        date(2025, 6, 20), date(2025, 6, 21), date(2025, 11, 1), date(2025, 12, 6),
        date(2025, 12, 24), date(2025, 12, 25), date(2025, 12, 26),
    }
    assert set(h) == expected
    assert h[date(2025, 6, 20)] == "Juhannusaatto"
    assert F.finnish_holidays(2026)[date(2026, 6, 19)] == "Juhannusaatto"
    assert date(2026, 10, 31) in F.finnish_holidays(2026)  # All Saints' Day (Sat)


def test_holiday_flag_uses_helsinki_date():
    # 2025-12-23 22:30 UTC is already Christmas Eve 00:30 in Helsinki
    ts = pd.DatetimeIndex(["2025-12-23 21:30", "2025-12-23 22:30"], tz="UTC")
    assert list(F.is_finnish_holiday(ts)) == [False, True]


def test_price_known_until():
    # 12:00 Helsinki (11:00 Stockholm, winter) -> end of today's Stockholm day
    t = pd.Timestamp("2025-01-15 12:00", tz="Europe/Helsinki")
    assert F.price_known_until(t) == pd.Timestamp("2025-01-15 23:00", tz="UTC")
    # 14:30 Helsinki = 13:30 Stockholm -> end of next Stockholm day
    t = pd.Timestamp("2025-01-15 14:30", tz="Europe/Helsinki")
    assert F.price_known_until(t) == pd.Timestamp("2025-01-16 23:00", tz="UTC")
    # Summer time
    t = pd.Timestamp("2025-06-15 15:00", tz="Europe/Helsinki")
    assert F.price_known_until(t) == pd.Timestamp("2025-06-16 22:00", tz="UTC")
    # DST change day: day before 2025-03-30 (23h day)
    t = pd.Timestamp("2025-03-29 15:00", tz="Europe/Helsinki")
    assert F.price_known_until(t) == pd.Timestamp("2025-03-30 22:00", tz="UTC")
    assert F.last_known_price_hour(t) == pd.Timestamp("2025-03-30 21:00", tz="UTC")


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def test_training_frame_columns_and_shape(syn_db):
    X, y, h = F.build_training_frame(syn_db, SYN_START + pd.Timedelta(days=20),
                                     SYN_START + pd.Timedelta(days=50))
    assert list(X.columns) == F.FEATURE_COLUMNS
    assert X.index.names == ["issued_at", "target_ts"]
    assert len(X) == len(y) == len(h) > 0
    assert y.notna().all()
    assert (X["lead_hours"] >= 0).all()
    assert h.between(1, 168).all()
    issued = X.index.get_level_values("issued_at")
    assert (issued.tz_convert("Europe/Helsinki").hour == 12).all()
    per_issue = X.groupby(level="issued_at").size()
    assert per_issue.max() <= F.TRAIN_HORIZONS_PER_ISSUE
    # strongest drivers present, unavailable sources are NaN columns
    assert X["residual_load_mw"].notna().any()
    assert X[S.SOLAR_FORECAST_ENTSOE_MW].isna().all()
    assert X["fi_same_hour_lastweek"].notna().mean() > 0.9


def test_training_frame_is_deterministic(syn_db):
    a = F.build_training_frame(syn_db, SYN_START + pd.Timedelta(days=20), SYN_START + pd.Timedelta(days=30))
    b = F.build_training_frame(syn_db, SYN_START + pd.Timedelta(days=20), SYN_START + pd.Timedelta(days=30))
    pd.testing.assert_frame_equal(a[0], b[0])


def test_noise_grows_with_horizon(syn_db):
    s, e = SYN_START + pd.Timedelta(days=20), SYN_START + pd.Timedelta(days=50)
    clean, _ = F.build_full_frame(syn_db, s, e, noise=False)
    noisy, _ = F.build_full_frame(syn_db, s, e, noise=True)
    ratio = (noisy[S.TEMPERATURE_FORECAST_C] - clean[S.TEMPERATURE_FORECAST_C]).abs()
    h = clean["horizon_hours"]
    assert ratio[h <= 24].mean() < ratio[h >= 144].mean()
    # calendar and price lags are never perturbed
    pd.testing.assert_series_equal(clean["fi_lastday_mean"], noisy["fi_lastday_mean"])


def test_forecast_coverage_masks(syn_db):
    t = SYN_START + pd.Timedelta(days=40, hours=10)
    X = F.build_prediction_frame(syn_db, t)
    h = X["horizon_hours"]
    assert X.loc[h > 72, S.CONSUMPTION_FORECAST_MW].isna().all()
    assert X.loc[h <= 72, S.CONSUMPTION_FORECAST_MW].notna().all()
    assert X.loc[h > 60, S.LOAD_FORECAST_MW].isna().all()  # ENTSO-E only until end of D+1
    assert X[S.TEMPERATURE_FORECAST_C].notna().all()  # Open-Meteo covers the full week
    assert X[S.WIND_CAPACITY_MW].nunique() == 1  # capacity as of issue


def test_prediction_frame_columns(syn_db):
    t = pd.Timestamp("2025-02-10 14:30", tz="Europe/Helsinki")
    X = F.build_prediction_frame(syn_db, t)
    assert list(X.columns) == F.FEATURE_COLUMNS
    assert len(X) == 168
    assert X.index[0] == pd.Timestamp("2025-02-10 13:00", tz="UTC")
    assert (X.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert X["fi_lastday_mean"].notna().all()
    assert X["nuclear_actual_mean_24h"].notna().all()
    # D+1 already published at 14:30 Helsinki -> negative lead there
    ku = F.price_known_until(t)
    assert ((X.index < ku) == (X["lead_hours"] < 0)).all()


def test_missing_sources_tolerated(tmp_path):
    engine = create_engine("sqlite:///%s" % (tmp_path / "empty.db"))
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()
    X = F.build_prediction_frame(db, pd.Timestamp("2025-02-10 12:00", tz="UTC"))
    assert list(X.columns) == F.FEATURE_COLUMNS
    assert len(X) == 168
    assert X[F.PRICE_FEATURES].isna().all().all()
    db.close()


def test_snapshot_preferred(tmp_path):
    db = make_session(tmp_path, days=40)
    issued = pd.Timestamp("2025-01-30 10:00", tz="UTC")
    target = issued + pd.Timedelta(hours=20)  # after the last known price hour
    db.execute(ForecastSnapshot.__table__.insert(), [{
        "issued_at": issued.to_pydatetime(), "timestamp": target.to_pydatetime(),
        "series": S.WIND_FORECAST_MW, "value": 1234.0,
    }])
    db.commit()
    X = F.build_prediction_frame(db, issued)
    assert X.loc[target, S.WIND_FORECAST_MW] == 1234.0
    # training frame for the issue at 12:00 Helsinki that day uses it without noise
    full, _ = F.build_full_frame(db, issued - pd.Timedelta(days=1), issued + pd.Timedelta(days=2))
    row = full[(full["_issued_at"] == issued) & (full["_target_ts"] == target)]
    assert len(row) == 1
    assert row[S.WIND_FORECAST_MW].iloc[0] == 1234.0
    db.close()


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_prediction_frame_no_price_lookahead(tmp_path):
    prices, _, _ = synthetic_data(60)
    db = make_session(tmp_path, days=60, prices=prices)
    issued = pd.Timestamp("2025-02-10 12:00", tz="Europe/Helsinki")
    before = F.build_prediction_frame(db, issued)

    ku = F.price_known_until(issued)
    mutated = prices.copy()
    mutated.loc[mutated.index >= ku] = 9999.0
    write_prices(db, mutated)
    after = F.build_prediction_frame(db, issued)
    pd.testing.assert_frame_equal(before, after)

    # sanity: mutating the last known hour does change features
    mutated.loc[ku - pd.Timedelta(hours=1)] = 9999.0
    write_prices(db, mutated)
    changed = F.build_prediction_frame(db, issued)
    assert not changed["fi_lastday_mean"].equals(before["fi_lastday_mean"])
    db.close()


def test_training_frame_no_price_lookahead(tmp_path):
    prices, _, _ = synthetic_data(60)
    db = make_session(tmp_path, days=60, prices=prices)
    s, e = SYN_START + pd.Timedelta(days=20), SYN_START + pd.Timedelta(days=50)
    X0, y0, _ = F.build_training_frame(db, s, e)

    issue = X0.index.get_level_values("issued_at").unique()[10]
    ku = F.price_known_until(issue)
    mutated = prices.copy()
    mutated.loc[mutated.index >= ku] = mutated.loc[mutated.index >= ku] * 3 + 500
    write_prices(db, mutated)
    X1, y1, _ = F.build_training_frame(db, s, e)

    rows0 = X0.xs(issue, level="issued_at", drop_level=False)
    rows1 = X1.xs(issue, level="issued_at", drop_level=False)
    pd.testing.assert_frame_equal(rows0, rows1)
    assert not np.allclose(y0.loc[rows0.index], y1.loc[rows1.index])  # targets did change
    # every price lag column of this issue only depends on hours < known_until
    later = X1.index.get_level_values("issued_at") > issue + pd.Timedelta(days=1)
    assert not X0.loc[later, "fi_lastday_mean"].equals(X1.loc[later, "fi_lastday_mean"])
    db.close()


def test_lagged_actuals_use_only_past(tmp_path):
    db = make_session(tmp_path, days=40)
    issued = pd.Timestamp("2025-01-30 10:00", tz="UTC")
    before = F.build_prediction_frame(db, issued)
    db.execute(DriverData.__table__.update()
               .where(DriverData.series == S.NUCLEAR_ACTUAL_MW)
               .where(DriverData.timestamp >= (issued - pd.Timedelta(hours=1)).to_pydatetime())
               .values(value=0.0))
    db.commit()
    after = F.build_prediction_frame(db, issued)
    pd.testing.assert_frame_equal(before, after)
    db.close()


def test_prices_available_until_follows_stored_prices():
    issued = pd.Timestamp("2026-11-13 14:00", tz="UTC")  # 15:00 Stockholm: D+1 published
    known = F.price_known_until(issued)  # 2026-11-14 23:00 UTC
    hours = pd.date_range(pd.Timestamp("2026-11-10 00:00", tz="UTC"), known, freq="h", inclusive="left")
    full = pd.DataFrame({"FI": 50.0}, index=hours)
    assert F.prices_available_until(full, issued) == known

    # D+1 not fetched yet: anchor at the end of the last fully stored Stockholm day
    today_end = pd.Timestamp("2026-11-13 23:00", tz="UTC")
    stored = full[full.index < today_end]
    assert F.prices_available_until(stored, issued) == today_end

    # partially stored day snaps back to that day's start
    partial = full[full.index < today_end - pd.Timedelta(hours=5)]
    assert F.prices_available_until(partial, issued) == pd.Timestamp("2026-11-12 23:00", tz="UTC")
