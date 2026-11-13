"""Outage features: known unavailable capacity per target hour, as of the issue time.

For each forecast issue time only outage document revisions published at or
before it are used; per document the latest such revision wins and cancelled
documents are dropped. This mirrors what a forecaster could have read in the
market messages at that moment, so the features are leakage-safe in training.
"""

from typing import Dict

import numpy as np
import pandas as pd

NUCLEAR_PSR = "B14"
BORDERS = ("FI-EE", "FI-SE1", "FI-SE3")

OUTAGE_FEATURES = [
    "nuclear_unavailable_mw",
    "other_gen_unavailable_mw",
    "ic_fi_ee_unavailable_mw",
    "ic_fi_se1_unavailable_mw",
    "ic_fi_se3_unavailable_mw",
] + ["ic_" + b.lower().replace("-", "_") + "_reported_available_mw" for b in BORDERS]

_HOUR_NS = 3_600_000_000_000


def known_events(events: pd.DataFrame, issued_at: pd.Timestamp) -> pd.DataFrame:
    """Active event periods as known at ``issued_at`` (latest revision per document)."""
    if events.empty:
        return events
    seen = events[events["published_at"] <= issued_at]
    if seen.empty:
        return seen
    latest = seen.groupby("mrid")["revision"].transform("max")
    seen = seen[seen["revision"] == latest]
    return seen[seen["status"] != "cancelled"]


def _feature_of(row) -> str:
    if row.kind == "generation":
        return "nuclear_unavailable_mw" if row.psr_type == NUCLEAR_PSR else "other_gen_unavailable_mw"
    zones = set(row.border.replace("_", "-").upper().split("-"))
    for border in BORDERS:  # either direction maps to the same border
        if zones == set(border.split("-")):
            return "ic_" + border.lower().replace("-", "_") + "_unavailable_mw"
    return ""


def outage_features(
    events: pd.DataFrame, issues: pd.DatetimeIndex, target: pd.DatetimeIndex, issue_idx: np.ndarray
) -> Dict[str, np.ndarray]:
    """Arrays aligned with ``target`` (one row per (issue, target hour)).

    Generation notices for the same resource use their maximum unavailable
    capacity, then resources are summed per category. Transmission outages are summed
    per border and direction, and the border value is the larger direction, so a
    line reported once per direction is not counted twice.
    """
    n = len(target)
    out = {name: np.zeros(n) for name in OUTAGE_FEATURES}
    for name in OUTAGE_FEATURES:
        if name.endswith("reported_available_mw"):
            out[name][:] = np.nan
    if events is None or events.empty or n == 0:
        return out
    events = events.copy()
    events["feature"] = [_feature_of(r) for r in events.itertuples(index=False)]
    events["direction"] = np.where(events["kind"] == "transmission", events["border"], "")
    t_ns = target.asi8
    for i, issued in enumerate(issues):
        rows = np.flatnonzero(issue_idx == i)
        if len(rows) == 0:
            continue
        lo, hi = t_ns[rows].min(), t_ns[rows].max() + _HOUR_NS
        known = known_events(events, issued)
        known = known[known["feature"] != ""]
        known = known[(known["start"].astype("int64") < hi) & (known["end"].astype("int64") > lo)]
        if known.empty:
            continue
        hours = t_ns[rows]
        by_dir: Dict[tuple, np.ndarray] = {}
        for ev in known.itertuples(index=False):
            active = (hours >= ev.start.value - (ev.start.value % _HOUR_NS)) & (hours < ev.end.value)
            if not active.any():
                continue
            if ev.kind == "transmission" and pd.notna(ev.available_mw):
                # Minimum reported available capacity among active notices,
                # not an estimate of the total border transfer capacity.
                name = ev.feature.replace("unavailable_mw", "reported_available_mw")
                selected = rows[active]
                out[name][selected] = np.fmin(out[name][selected], float(ev.available_mw))
            if pd.isna(ev.unavailable_mw):
                continue
            key = (ev.feature, ev.direction, (ev.resource or ev.mrid) if ev.kind == "generation" else "")
            acc = by_dir.setdefault(key, np.zeros(len(rows)))
            if ev.kind == "generation":
                acc[active] = np.maximum(acc[active], float(ev.unavailable_mw))
            else:
                acc[active] += float(ev.unavailable_mw)
        per_feature: Dict[str, np.ndarray] = {}
        for (feature, _direction, _resource), acc in by_dir.items():
            prev = per_feature.get(feature)
            if prev is None:
                per_feature[feature] = acc
            elif feature.startswith("ic_"):  # other direction of the same border
                per_feature[feature] = np.maximum(prev, acc)
            else:
                per_feature[feature] = prev + acc
        for feature, vals in per_feature.items():
            out[feature][rows] = vals
    return out
