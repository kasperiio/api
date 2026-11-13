"""Feature pipeline: turns stored drivers and prices into a model matrix.

Rows are (issue time, target hour) pairs. Every frame returned here has exactly
``FEATURE_COLUMNS`` (in that order); inputs that are unavailable become NaN
columns, which LightGBM handles natively.

Information sets
----------------
* Prices: only hours with timestamp < ``price_known_until(issued_at)`` are used
  (day-ahead prices for D+1 are published ~13:00 CET; we are conservative).
* ``*_forecast_*`` drivers: value for the target hour (a forecast is knowable).
  In training these come from ``driver_data`` which is near-actual history, so
  horizon-dependent noise emulates forecast error unless a forecast snapshot
  captured at/before the issue time exists (then the snapshot is used as is).
* ``*_actual_*`` drivers: only as means over the 24 h ending 1 h before issue.
* Hydro reservoirs: weekly, stamped at week start, published 1-2 weeks after the
  week ends -> value stamped at/before issue - 21 days, plus its 1-week change.
* Wind capacity: stepwise; last known value at the issue hour for all targets.
* Live forecasts do not cover 7 days (Fingrid ~72 h, ENTSO-E until end of
  D+1), so those series are masked to NaN beyond ``SERIES_MAX_HORIZON`` in both
  training and prediction; Open-Meteo weather carries the long horizons.
* Open-Meteo radiation is period-end stamped and shifted 1 h back on load.
* Archived weather overrides simulated values only when its model run is at
  least six hours older than the issue; captured snapshots take precedence.
* Outages use the latest document revision published by issue time. Carbon
  auction prices enter from the next UTC day and expire after seven days.

Training issue times are one per day at 12:00 Europe/Helsinki. Only target
hours whose price is not yet known at issue time (lead_hours >= 0) and within
168 h are used, and each issue is subsampled to ``TRAIN_HORIZONS_PER_ISSUE``
random target hours (the first unknown 24 h are always kept) to keep training
fast; daily aggregates are computed on the full grid before subsampling.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.forecasting import series as S
from app.forecasting import store
from app.forecasting.outages import OUTAGE_FEATURES, outage_features

logger = logging.getLogger(__name__)

HELSINKI = "Europe/Helsinki"
STOCKHOLM = "Europe/Stockholm"
ISSUE_HOUR_LOCAL = 12  # training issue time, Europe/Helsinki
MAX_HORIZON_HOURS = 168
TRAIN_HORIZONS_PER_ISSUE = 72
NOISE_SEED = 20230101
# Weekly reservoir values are stamped at week start and published 1-2 weeks after
# the week ends -> conservative availability lag of 3 weeks from the stamp.
RESERVOIR_DELAY = pd.Timedelta(days=21)
LOOKBACK = pd.Timedelta(days=45)

# Driver series looked up at the target hour.
FORECAST_SERIES = [
    S.WIND_FORECAST_MW,
    S.SOLAR_FORECAST_MW,
    S.CONSUMPTION_FORECAST_MW,
    S.PRODUCTION_FORECAST_MW,
    S.TEMPERATURE_FORECAST_C,
    S.WIND_SPEED_100M_FORECAST_MS,
    S.SOLAR_RADIATION_FORECAST_WM2,
    S.LOAD_FORECAST_MW,
    S.WIND_FORECAST_ENTSOE_MW,
    S.SOLAR_FORECAST_ENTSOE_MW,
] + list(S.TRANSFER_CAPACITY_SERIES)
RESERVOIR_SERIES = [S.HYDRO_RESERVOIR_NO_MWH, S.HYDRO_RESERVOIR_SE_MWH]
# Stepwise series: last known value at the issue hour is used for every target hour.
ISSUE_SERIES = [S.WIND_CAPACITY_MW, S.EUA_AUCTION_EUR_T]
ACTUAL_SERIES = [S.NUCLEAR_ACTUAL_MW, S.NET_IMPORT_ACTUAL_MW, S.HYDRO_ACTUAL_MW]
FFILL_SERIES = ISSUE_SERIES + RESERVOIR_SERIES
ALL_SERIES = FORECAST_SERIES + ISSUE_SERIES + RESERVOIR_SERIES + ACTUAL_SERIES

# Live forecast coverage. History in driver_data has every hour, but live
# forecasts do not reach 7 days, so series are masked to NaN beyond what the
# live source covers (in training AND prediction) to avoid train/serve skew.
# int = max hours after the issue hour; "D+1" = until end of the next
# Stockholm/CET day after the issue; absent = full horizon (Open-Meteo, 16 d).
D_PLUS_1 = "D+1"
SERIES_MAX_HORIZON = {
    S.CONSUMPTION_FORECAST_MW: 72,
    S.PRODUCTION_FORECAST_MW: 72,
    S.WIND_FORECAST_MW: 72,
    S.SOLAR_FORECAST_MW: 72,
    S.LOAD_FORECAST_MW: D_PLUS_1,
    S.WIND_FORECAST_ENTSOE_MW: D_PLUS_1,
    S.SOLAR_FORECAST_ENTSOE_MW: D_PLUS_1,
}
SERIES_MAX_HORIZON.update({name: D_PLUS_1 for name in S.TRANSFER_CAPACITY_SERIES})
# Open-Meteo radiation at hour H is the mean over (H-1, H]; shift to period start.
PERIOD_END_SERIES = [S.SOLAR_RADIATION_FORECAST_WM2]

# Forecast-error emulation: (series, group, scale within group)
# group -> (kind, sigma at h<=24, sigma at h=168)
NOISE_GROUPS = {
    "wind": ("mult", 0.10, 0.35),
    "solar": ("mult", 0.15, 0.40),
    "cons": ("mult", 0.02, 0.06),
    "temp": ("add", 0.5, 3.0),
}
NOISE_SERIES = {
    S.WIND_FORECAST_MW: ("wind", 1.0),
    S.WIND_FORECAST_ENTSOE_MW: ("wind", 1.0),
    S.WIND_SPEED_100M_FORECAST_MS: ("wind", 0.4),  # power ~ speed^3
    S.SOLAR_FORECAST_MW: ("solar", 1.0),
    S.SOLAR_FORECAST_ENTSOE_MW: ("solar", 1.0),
    S.SOLAR_RADIATION_FORECAST_WM2: ("solar", 1.0),
    S.CONSUMPTION_FORECAST_MW: ("cons", 1.0),
    S.LOAD_FORECAST_MW: ("cons", 1.0),
    S.PRODUCTION_FORECAST_MW: ("cons", 1.0),
    S.TEMPERATURE_FORECAST_C: ("temp", 1.0),
}
NOISE_AR_PHI = 0.97  # hourly autocorrelation of forecast errors

CALENDAR_FEATURES = [
    "hour", "dow", "month", "is_holiday", "is_offday", "horizon_hours", "lead_hours",
]
DERIVED_FEATURES = [
    "residual_load_mw", "wind_share", "wind_cf",
    "residual_load_day_mean", "residual_load_day_max", "wind_day_mean",
    "temperature_day_mean", "wind_speed_day_mean", "residual_load_lastday_mean", "residual_load_delta",
]
PRICE_FEATURES = [
    "fi_lastday_mean", "fi_lastday_min", "fi_lastday_max",
    "fi_same_hour_lastday", "fi_same_hour_lastweek",
    "fi_mean_7d", "fi_std_7d",
    "se1_lastday_mean", "se3_lastday_mean", "ee_lastday_mean", "fi_se1_spread_lastday",
]
RESERVOIR_CHANGE_FEATURES = ["hydro_reservoir_no_change_1w", "hydro_reservoir_se_change_1w"]
ACTUAL_FEATURES = [
    "nuclear_actual_mean_24h", "net_import_actual_mean_24h", "hydro_actual_mean_24h",
]
FEATURE_COLUMNS: List[str] = (
    CALENDAR_FEATURES + FORECAST_SERIES + ISSUE_SERIES + RESERVOIR_SERIES
    + RESERVOIR_CHANGE_FEATURES + DERIVED_FEATURES
    + PRICE_FEATURES + ACTUAL_FEATURES + OUTAGE_FEATURES
)


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

def easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def finnish_holidays(year: int) -> Dict[date, str]:
    """Finnish public holidays plus de facto days off (Midsummer Eve, Christmas Eve)."""
    easter = easter_sunday(year)
    midsummer_eve = date(year, 6, 19) + timedelta(days=(4 - date(year, 6, 19).weekday()) % 7)
    all_saints = date(year, 10, 31) + timedelta(days=(5 - date(year, 10, 31).weekday()) % 7)
    return {
        date(year, 1, 1): "Uudenvuodenpäivä",
        date(year, 1, 6): "Loppiainen",
        easter - timedelta(days=2): "Pitkäperjantai",
        easter: "Pääsiäispäivä",
        easter + timedelta(days=1): "2. pääsiäispäivä",
        date(year, 5, 1): "Vappu",
        easter + timedelta(days=39): "Helatorstai",
        easter + timedelta(days=49): "Helluntaipäivä",
        midsummer_eve: "Juhannusaatto",
        midsummer_eve + timedelta(days=1): "Juhannuspäivä",
        all_saints: "Pyhäinpäivä",
        date(year, 12, 6): "Itsenäisyyspäivä",
        date(year, 12, 24): "Jouluaatto",
        date(year, 12, 25): "Joulupäivä",
        date(year, 12, 26): "Tapaninpäivä",
    }


def is_finnish_holiday(ts: pd.DatetimeIndex) -> np.ndarray:
    """Boolean array: does each (tz-aware) timestamp fall on a Finnish holiday (Helsinki date)."""
    local = pd.DatetimeIndex(ts).tz_convert(HELSINKI).tz_localize(None).normalize()
    if len(local) == 0:
        return np.zeros(0, dtype=bool)
    hol = []
    for y in range(local.min().year, local.max().year + 1):
        hol.extend(finnish_holidays(y).keys())
    return local.isin(pd.DatetimeIndex(hol))


def price_known_until(issued_at) -> pd.Timestamp:
    """Exclusive UTC end of the prices known at ``issued_at``.

    Day-ahead prices for the next day are published ~13:00 CET. If the
    Stockholm-local issue hour is >= 13 prices are known until the end of the
    next Stockholm day, otherwise until the end of the current one.
    Features may only use prices with timestamp < this value.
    """
    ts = pd.Timestamp(issued_at)
    if ts.tzinfo is None:
        raise ValueError("issued_at must be timezone-aware")
    days = 2 if ts.tz_convert(STOCKHOLM).hour >= 13 else 1
    return _stockholm_day_end(ts, days)


def _stockholm_day_end(ts: pd.Timestamp, days: int) -> pd.Timestamp:
    """UTC instant of Stockholm-local midnight ``days`` days after ts's local date."""
    local = pd.Timestamp(ts).tz_convert(STOCKHOLM)
    end_naive = local.tz_localize(None).normalize() + pd.Timedelta(days=days)
    return end_naive.tz_localize(STOCKHOLM).tz_convert("UTC")


def prices_available_until(prices: pd.DataFrame, issued_at) -> pd.Timestamp:
    """Exclusive end of the FI prices actually present in ``prices`` (hourly frame),
    capped by ``price_known_until(issued_at)``. Snapped down to a whole Stockholm
    day boundary so the lags look like those of an earlier issue time, which the
    model was trained on."""
    known = price_known_until(issued_at)
    if "FI" not in prices.columns:
        return known
    fi = prices["FI"].dropna()
    fi = fi[fi.index < known]
    if fi.empty:
        return known
    last = fi.index.max() + pd.Timedelta(hours=1)
    if last >= known:
        return known
    day_start = last.tz_convert(STOCKHOLM).normalize().tz_convert("UTC")
    return day_start if day_start < last else last


def last_known_price_hour(issued_at) -> pd.Timestamp:
    """UTC start of the last hour whose price is known at ``issued_at``."""
    return price_known_until(issued_at) - pd.Timedelta(hours=1)


def training_issue_times(start, end) -> pd.DatetimeIndex:
    """Daily 12:00 Europe/Helsinki issue times (UTC) whose 168 h window overlaps [start, end)."""
    start, end = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
    first = (start - pd.Timedelta(days=8)).tz_convert(HELSINKI).tz_localize(None).normalize()
    last = end.tz_convert(HELSINKI).tz_localize(None).normalize()
    days = pd.date_range(first, last, freq="D")
    issues = (days + pd.Timedelta(hours=ISSUE_HOUR_LOCAL)).tz_localize(HELSINKI).tz_convert("UTC")
    keep = (issues < end) & (issues + pd.Timedelta(hours=MAX_HORIZON_HOURS) >= start)
    return issues[keep]


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------

def _hour_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.floor("h"), end.ceil("h"), freq="h", tz="UTC", inclusive="left")


def _load_prices(db: Session, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    p = store.load_spot_prices(db, start, end, zones=S.ZONES)
    p.index = pd.DatetimeIndex(p.index)
    p = p.reindex(columns=list(S.ZONES))
    return p.reindex(_hour_range(start, end)).astype(float)


def _load_drivers(db: Session, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # +1 h so period-end series can be shifted back into the last hour
    d = store.load_drivers(db, start, end + pd.Timedelta(hours=1))
    d.index = pd.DatetimeIndex(d.index)
    d = _shift_period_end(d.reindex(columns=ALL_SERIES))
    d = d.reindex(_hour_range(start, end)).astype(float)
    carbon = d[S.EUA_AUCTION_EUR_T].copy()
    d[FFILL_SERIES] = d[FFILL_SERIES].ffill()
    # Carbon auctions pause over holidays and summers. Do not silently carry
    # a stale market price indefinitely across an archive gap.
    d[S.EUA_AUCTION_EUR_T] = carbon.ffill(limit=7 * 24)
    return d


def _shift_period_end(frame: pd.DataFrame) -> pd.DataFrame:
    """Move period-end stamped series one hour back (value at H -> H-1)."""
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    for name in PERIOD_END_SERIES:
        if name in frame.columns and frame[name].notna().any():
            col = frame[name].dropna()
            col.index = col.index - pd.Timedelta(hours=1)
            frame[name] = col.reindex(frame.index)
    return frame


def _as_wide(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    if any(n in frame.columns for n in PERIOD_END_SERIES):
        full = frame.index.union(frame.index - pd.Timedelta(hours=1))
        frame = _shift_period_end(frame.reindex(full))
    return frame.reindex(columns=[c for c in FORECAST_SERIES if c in frame.columns]).astype(float)


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------

def _noise_factors(issues: pd.DatetimeIndex, seed: int) -> Dict[str, np.ndarray]:
    """AR(1) standard-normal error paths per issue & group, shape (n_issue, H).

    Each issue gets its own RNG seeded by (seed, issue hour) so the noise for an
    issue does not depend on the requested date range.
    """
    groups = list(NOISE_GROUPS)
    n, H, G = len(issues), MAX_HORIZON_HOURS, len(groups)
    z = np.empty((n, H, G))
    hours = (issues.asi8 // 3_600_000_000_000).astype(np.int64)
    for i, hr in enumerate(hours):
        z[i] = np.random.default_rng([seed, int(hr) % (2 ** 32)]).standard_normal((H, G))
    e = np.empty_like(z)
    e[:, 0] = z[:, 0]
    a = np.sqrt(1 - NOISE_AR_PHI ** 2)
    for h in range(1, H):
        e[:, h] = NOISE_AR_PHI * e[:, h - 1] + a * z[:, h]
    h_axis = np.arange(1, H + 1)
    out = {}
    for gi, g in enumerate(groups):
        _, s0, s1 = NOISE_GROUPS[g]
        sigma = np.interp(h_axis, [0, 24, H], [s0, s0, s1])
        out[g] = e[:, :, gi] * sigma[None, :]
    return out


def _assemble(
    prices: pd.DataFrame,
    drivers: pd.DataFrame,
    issues: pd.DatetimeIndex,
    noise: bool,
    seed: int,
    snapshots: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None,
    known_until: Optional[pd.DatetimeIndex] = None,
    archived_weather: Optional[pd.DataFrame] = None,
    outages: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Full grid (each issue x horizons 1..168) feature frame.

    ``known_until`` (one per issue) overrides ``price_known_until`` when fewer
    prices are actually available than the publication schedule implies.

    Returns a frame with FEATURE_COLUMNS plus helper columns ``_issued_at``,
    ``_target_ts`` and ``_issue_idx``. ``prices`` must only contain what is
    allowed to be seen (rolling stats look strictly backwards from the known end).
    """
    H = MAX_HORIZON_HOURS
    n = len(issues)
    base = issues.floor("h")
    h_arr = np.tile(np.arange(1, H + 1), n)
    issue_idx = np.repeat(np.arange(n), H)
    target = pd.DatetimeIndex(np.repeat(base.asi8, H) + h_arr * 3_600_000_000_000, tz="UTC")
    if known_until is None:
        known_until = pd.DatetimeIndex([price_known_until(t) for t in issues])
    ku_rows = known_until[issue_idx]
    lead = ((target.asi8 - ku_rows.asi8) // 3_600_000_000_000).astype(float)

    cols: Dict[str, np.ndarray] = {}

    # Calendar
    loc = target.tz_convert(HELSINKI)
    cols["hour"] = loc.hour.values.astype(float)
    cols["dow"] = loc.dayofweek.values.astype(float)
    cols["month"] = loc.month.values.astype(float)
    hol = is_finnish_holiday(target)
    cols["is_holiday"] = hol.astype(float)
    cols["is_offday"] = (hol | (loc.dayofweek.values >= 5)).astype(float)
    cols["horizon_hours"] = h_arr.astype(float)
    cols["lead_hours"] = lead

    # Forecast drivers at target hour
    fvals = drivers[FORECAST_SERIES].reindex(target).to_numpy(copy=True)
    if noise:
        factors = _noise_factors(issues, seed)
        for j, name in enumerate(FORECAST_SERIES):
            if name not in NOISE_SERIES:
                continue
            g, scale = NOISE_SERIES[name]
            err = factors[g].reshape(-1) * scale
            if NOISE_GROUPS[g][0] == "mult":
                fvals[:, j] = fvals[:, j] * np.exp(err - 0.5 * err ** 2)
            else:
                fvals[:, j] = fvals[:, j] + err
    # Choose an archive age whose model run predates the issue by at least six
    # hours. Previous Runs has no exact publication timestamps; using a younger
    # run would let future weather information into historical evaluation.
    if archived_weather is not None and not archived_weather.empty:
        archive = archived_weather.copy()
        archive["timestamp"] = archive["timestamp"] - archive["series"].isin(
            PERIOD_END_SERIES).astype(int) * pd.Timedelta(hours=1)
        for j, name in enumerate(FORECAST_SERIES):
            sub = archive[archive["series"] == name]
            if sub.empty:
                continue
            values = sub.set_index(["timestamp", "lead_days"])["value"]
            ages = np.ceil((h_arr + 6 + int(name in PERIOD_END_SERIES)) / 24).astype(int)
            keys = pd.MultiIndex.from_arrays([target, ages])
            v = values.reindex(keys).to_numpy()
            m = ~np.isnan(v)
            fvals[m, j] = v[m]
    if snapshots:
        for i, snap in snapshots.items():
            if snap is None or snap.empty:
                continue
            sl = slice(i * H, (i + 1) * H)
            sw = _as_wide(snap).reindex(target[sl])
            for name in sw.columns:
                j = FORECAST_SERIES.index(name)
                v = sw[name].to_numpy()
                m = ~np.isnan(v)
                fvals[sl, j][m] = v[m]
    # Mask beyond live forecast coverage
    d1_end = pd.DatetimeIndex([_stockholm_day_end(t, 2) for t in issues])
    for j, name in enumerate(FORECAST_SERIES):
        lim = SERIES_MAX_HORIZON.get(name)
        if lim is None:
            continue
        if lim == D_PLUS_1:
            fvals[target.asi8 >= d1_end.asi8[issue_idx], j] = np.nan
        else:
            fvals[h_arr > lim, j] = np.nan
    for j, name in enumerate(FORECAST_SERIES):
        cols[name] = fvals[:, j]
    cols.update(outage_features(outages, issues, target, issue_idx))

    # Stepwise series (capacity): last known value at the issue hour
    for name in ISSUE_SERIES:
        cols[name] = drivers[name].reindex(base).to_numpy()[issue_idx]

    # Reservoirs: as of issue - delay
    res_t = (issues - RESERVOIR_DELAY).floor("h")
    res_prev = res_t - pd.Timedelta(days=7)
    for name, chg in zip(RESERVOIR_SERIES, RESERVOIR_CHANGE_FEATURES):
        now_v = drivers[name].reindex(res_t).to_numpy()
        cols[name] = now_v[issue_idx]
        cols[chg] = (now_v - drivers[name].reindex(res_prev).to_numpy())[issue_idx]

    # Derived
    def first_valid(*arrs):
        out = arrs[0].copy()
        for a in arrs[1:]:
            out = np.where(np.isnan(out), a, out)
        return out

    cons = first_valid(cols[S.CONSUMPTION_FORECAST_MW], cols[S.LOAD_FORECAST_MW])
    wind = first_valid(cols[S.WIND_FORECAST_MW], cols[S.WIND_FORECAST_ENTSOE_MW])
    solar = first_valid(cols[S.SOLAR_FORECAST_MW], cols[S.SOLAR_FORECAST_ENTSOE_MW])
    solar = np.where(np.isnan(solar), 0.0, solar)
    resid = cons - wind - solar
    cols["residual_load_mw"] = resid
    with np.errstate(divide="ignore", invalid="ignore"):
        cols["wind_share"] = np.where(cons > 0, wind / cons, np.nan)
        cap = cols[S.WIND_CAPACITY_MW]
        cols["wind_cf"] = np.where(cap > 0, wind / cap, np.nan)

    day_key = pd.DataFrame({
        "i": issue_idx,
        "d": loc.tz_localize(None).normalize().asi8,
        "r": resid,
        "w": wind,
        "t": cols[S.TEMPERATURE_FORECAST_C],
        "ws": cols[S.WIND_SPEED_100M_FORECAST_MS],
    })
    grp = day_key.groupby(["i", "d"], sort=False)
    cols["residual_load_day_mean"] = grp["r"].transform("mean").to_numpy()
    cols["residual_load_day_max"] = grp["r"].transform("max").to_numpy()
    cols["wind_day_mean"] = grp["w"].transform("mean").to_numpy()
    cols["temperature_day_mean"] = grp["t"].transform("mean").to_numpy()
    cols["wind_speed_day_mean"] = grp["ws"].transform("mean").to_numpy()

    # Residual load over the last known price day (un-noised driver history)
    d_cons = drivers[S.CONSUMPTION_FORECAST_MW].fillna(drivers[S.LOAD_FORECAST_MW])
    d_wind = drivers[S.WIND_FORECAST_MW].fillna(drivers[S.WIND_FORECAST_ENTSOE_MW])
    d_solar = drivers[S.SOLAR_FORECAST_MW].fillna(drivers[S.SOLAR_FORECAST_ENTSOE_MW]).fillna(0.0)
    d_resid = (d_cons - d_wind - d_solar).rolling(24, min_periods=12).mean()
    last_hour = known_until - pd.Timedelta(hours=1)
    rl_last = d_resid.reindex(last_hour).to_numpy()[issue_idx]
    cols["residual_load_lastday_mean"] = rl_last
    cols["residual_load_delta"] = cols["residual_load_day_mean"] - rl_last

    # Price lags (prices strictly before known_until)
    fi = prices["FI"]
    r24 = fi.rolling(24, min_periods=12)
    r168 = fi.rolling(168, min_periods=72)
    stats = pd.DataFrame({
        "fi_lastday_mean": r24.mean(),
        "fi_lastday_min": r24.min(),
        "fi_lastday_max": r24.max(),
        "fi_mean_7d": r168.mean(),
        "fi_std_7d": r168.std(),
        "se1_lastday_mean": prices["SE1"].rolling(24, min_periods=12).mean(),
        "se3_lastday_mean": prices["SE3"].rolling(24, min_periods=12).mean(),
        "ee_lastday_mean": prices["EE"].rolling(24, min_periods=12).mean(),
    }).reindex(last_hour)
    for c in stats.columns:
        cols[c] = stats[c].to_numpy()[issue_idx]
    cols["fi_se1_spread_lastday"] = cols["fi_lastday_mean"] - cols["se1_lastday_mean"]

    # Same hour on the last known day / week: largest k with target - k*period < known_until
    k_day = np.maximum(1, np.ceil((lead + 1) / 24.0)).astype(np.int64)
    k_week = np.maximum(1, np.ceil((lead + 1) / 168.0)).astype(np.int64)
    hr_ns = 3_600_000_000_000
    lag_d = pd.DatetimeIndex(target.asi8 - k_day * 24 * hr_ns, tz="UTC")
    lag_w = pd.DatetimeIndex(target.asi8 - k_week * 168 * hr_ns, tz="UTC")
    cols["fi_same_hour_lastday"] = fi.reindex(lag_d).to_numpy()
    cols["fi_same_hour_lastweek"] = fi.reindex(lag_w).to_numpy()

    # Lagged actuals: mean over [I-25h, I-1h)
    act_t = base - pd.Timedelta(hours=2)
    for name, feat in zip(ACTUAL_SERIES, ACTUAL_FEATURES):
        roll = drivers[name].rolling(24, min_periods=6).mean()
        cols[feat] = roll.reindex(act_t).to_numpy()[issue_idx]

    X = pd.DataFrame({c: np.asarray(cols[c], dtype=float) for c in FEATURE_COLUMNS})
    X["_issued_at"] = issues[issue_idx]
    X["_target_ts"] = target
    X["_issue_idx"] = issue_idx
    return X


def _load_snapshots(
    db: Session, issues: pd.DatetimeIndex
) -> Dict[int, pd.DataFrame]:
    """Snapshots as of each issue time (only queried where any snapshot can apply)."""
    if len(issues) == 0:
        return {}
    win = pd.Timedelta(hours=MAX_HORIZON_HOURS + 1)
    probe = store.load_snapshot_asof(db, issues.max(), issues.min(), issues.max() + win)
    if probe.empty:
        return {}
    probe_idx = pd.DatetimeIndex(probe.index)
    out = {}
    for i, t in enumerate(issues):
        if ((probe_idx > t) & (probe_idx <= t + win)).any():
            snap = store.load_snapshot_asof(db, t, t, t + win)
            if not snap.empty:
                out[i] = snap
    if out:
        logger.info("Using forecast snapshots for %d/%d issue times", len(out), len(issues))
    return out


def build_full_frame(
    db: Session, start: datetime, end: datetime, noise: bool = True, seed: int = NOISE_SEED
) -> Tuple[pd.DataFrame, pd.Series]:
    """Full-horizon training-style frame for daily 12:00 issues with targets in [start, end).

    Returns (frame, y): ``frame`` has FEATURE_COLUMNS plus ``_issued_at``,
    ``_target_ts``, ``_issue_idx``; only rows with lead_hours >= 0 and a known
    FI price are kept. Used by build_training_frame and the walk-forward backtest.
    """
    start, end = pd.Timestamp(start).tz_convert("UTC"), pd.Timestamp(end).tz_convert("UTC")
    issues = training_issue_times(start, end)
    if len(issues) == 0:
        empty = pd.DataFrame(columns=FEATURE_COLUMNS + ["_issued_at", "_target_ts", "_issue_idx"])
        return empty, pd.Series(dtype=float)
    lo = issues.min() - LOOKBACK
    hi = issues.max() + pd.Timedelta(hours=MAX_HORIZON_HOURS + 2)
    prices = _load_prices(db, lo, hi)
    drivers = _load_drivers(db, lo, hi)
    snaps = _load_snapshots(db, issues)
    archive = store.load_archived_weather(db, lo, hi)
    events = store.load_outages(db, lo, hi, published_before=issues.max())
    X = _assemble(prices, drivers, issues, noise=noise, seed=seed, snapshots=snaps,
                  archived_weather=archive, outages=events)
    y = prices["FI"].reindex(pd.DatetimeIndex(X["_target_ts"])).to_numpy()
    keep = (
        (X["lead_hours"].to_numpy() >= 0)
        & (X["_target_ts"] >= start).to_numpy()
        & (X["_target_ts"] < end).to_numpy()
        & ~np.isnan(y)
    )
    X = X.loc[keep].reset_index(drop=True)
    return X, pd.Series(y[keep], index=X.index, name="price")


def subsample_horizons(frame: pd.DataFrame, per_issue: int = TRAIN_HORIZONS_PER_ISSUE,
                       seed: int = NOISE_SEED) -> np.ndarray:
    """Boolean mask keeping the first 24 unknown hours plus random others, ``per_issue`` per issue."""
    lead = frame["lead_hours"].to_numpy()
    issue_hours = (pd.DatetimeIndex(frame["_issued_at"]).asi8 // 3_600_000_000_000)
    rng = np.random.default_rng(seed + 1)
    score = rng.random(len(frame))
    score[lead < 24] = -1.0  # always keep
    rank = pd.Series(score).groupby(issue_hours).rank(method="first").to_numpy()
    return rank <= per_issue


def _to_xy_index(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(frame["_issued_at"]), pd.DatetimeIndex(frame["_target_ts"])],
        names=["issued_at", "target_ts"],
    )


def build_training_frame(
    db: Session, start: datetime, end: datetime, noise: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, horizon_hours) for training on target hours in [start, end).
    Rows are (issue time, target hour) pairs covering horizons 1..168 h; y is FI price
    (EUR/MWh). ``noise`` perturbs forecast drivers that came from near-actual history.

    X has exactly FEATURE_COLUMNS and a MultiIndex (issued_at, target_ts), both UTC;
    y and horizon_hours share that index. Issues are daily 12:00 Europe/Helsinki,
    targets restricted to hours whose price is unknown at issue time and subsampled
    to TRAIN_HORIZONS_PER_ISSUE per issue."""
    frame, y = build_full_frame(db, start, end, noise=noise)
    if len(frame):
        mask = subsample_horizons(frame)
        frame, y = frame.loc[mask], y.loc[mask]
    idx = _to_xy_index(frame)
    X = frame[FEATURE_COLUMNS].set_axis(idx)
    y = pd.Series(y.to_numpy(), index=idx, name="price")
    return X, y, X["horizon_hours"].rename("horizon_hours")


def build_prediction_frame(
    db: Session, issued_at: datetime, horizon_days: int = 7
) -> pd.DataFrame:
    """Feature matrix X indexed by UTC target hour for issued_at..issued_at+horizon_days,
    using only information available at ``issued_at`` (snapshots preferred).

    Target hours are the whole hours after ``issued_at`` (horizon_hours = 1..24*horizon_days
    from the floored issue hour). Rows with lead_hours < 0 are hours whose price is already
    published; callers should prefer the actual price there (see price_known_until).

    If the latest FI prices are not stored yet (publication late, refresh not
    run), price lags anchor on the last stored hour instead, the same way they
    would at an earlier issue time; see ``prices_available_until``."""
    issued = pd.Timestamp(issued_at)
    if issued.tzinfo is None:
        raise ValueError("issued_at must be timezone-aware")
    issued = issued.tz_convert("UTC")
    issues = pd.DatetimeIndex([issued])
    lo = issued.floor("h") - LOOKBACK
    hi = issued.floor("h") + pd.Timedelta(hours=MAX_HORIZON_HOURS + 2)
    prices = _load_prices(db, lo, price_known_until(issued))  # never load prices not yet known
    known = prices_available_until(prices, issued)
    prices = prices[prices.index < known].reindex(_hour_range(lo, hi))
    drivers = _load_drivers(db, lo, hi)
    snap = store.load_snapshot_asof(db, issued, issued.floor("h"), hi)
    snaps = {0: snap} if not snap.empty else None
    X = _assemble(prices, drivers, issues, noise=False, seed=NOISE_SEED, snapshots=snaps,
                  known_until=pd.DatetimeIndex([known]),
                  outages=store.load_outages(db, issued, hi, published_before=issued))
    n = min(int(horizon_days) * 24, MAX_HORIZON_HOURS)
    X = X.iloc[:n]
    out = X[FEATURE_COLUMNS].copy()
    out.index = pd.DatetimeIndex(X["_target_ts"], name="target_ts")
    return out
