import numpy as np
import pandas as pd

from app.forecasting.outages import outage_features

T0 = pd.Timestamp("2025-03-10 00:00", tz="UTC")


def ts(hours):
    return T0 + pd.Timedelta(hours=hours)


def event(mrid, revision, published, start, end, mw, kind="generation", psr="B14",
          border="", status="active"):
    return dict(mrid=mrid, revision=revision, published_at=ts(published), kind=kind,
                business_type="planned", status=status, resource=mrid, border=border, psr_type=psr,
                nominal_mw=np.nan, start=ts(start), end=ts(end), available_mw=np.nan,
                unavailable_mw=float(mw))


def grid(issue_hours, horizon=48):
    issues = pd.DatetimeIndex([ts(h) for h in issue_hours])
    target = pd.DatetimeIndex([i + pd.Timedelta(hours=k) for i in issues for k in range(1, horizon + 1)])
    issue_idx = np.repeat(np.arange(len(issues)), horizon)
    return issues, target, issue_idx


def test_only_revisions_published_before_issue_are_used():
    events = pd.DataFrame([
        event("ol3", 1, published=-24, start=10, end=20, mw=1600),
        event("ol3", 2, published=5, start=10, end=30, mw=1600),  # extension published later
    ])
    issues, target, idx = grid([0, 6])
    nuc = outage_features(events, issues, target, idx)["nuclear_unavailable_mw"]
    first, second = nuc[:48], nuc[48:]
    hours_first = (target[:48] - T0) / pd.Timedelta(hours=1)
    assert first[(hours_first >= 10) & (hours_first < 20)].min() == 1600
    assert first[hours_first >= 20].max() == 0  # extension not known yet
    hours_second = (target[48:] - T0) / pd.Timedelta(hours=1)
    assert second[(hours_second >= 20) & (hours_second < 30)].min() == 1600


def test_unknown_future_outage_is_not_visible():
    events = pd.DataFrame([event("forced", 1, published=12, start=12, end=40, mw=800)])
    issues, target, idx = grid([0])
    assert outage_features(events, issues, target, idx)["nuclear_unavailable_mw"].max() == 0


def test_cancelled_revision_removes_outage():
    events = pd.DataFrame([
        event("ol2", 1, published=-10, start=5, end=15, mw=890),
        event("ol2", 2, published=-5, start=5, end=15, mw=890, status="cancelled"),
    ])
    issues, target, idx = grid([0])
    assert outage_features(events, issues, target, idx)["nuclear_unavailable_mw"].max() == 0


def test_transmission_directions_not_double_counted_and_generation_summed():
    events = pd.DataFrame([
        event("estlink-a", 1, -1, 0, 48, 650, kind="transmission", psr="", border="FI-EE"),
        event("estlink-b", 1, -1, 0, 48, 650, kind="transmission", psr="", border="EE-FI"),
        event("chp1", 1, -1, 0, 48, 100, psr="B05"),
        event("chp2", 1, -1, 0, 48, 50, psr="B04"),
    ])
    issues, target, idx = grid([0])
    feats = outage_features(events, issues, target, idx)
    assert feats["other_gen_unavailable_mw"].max() == 150
    assert feats["ic_fi_ee_unavailable_mw"].max() == 650


def test_reported_transmission_capacity_does_not_invent_nominal_rating():
    row = event("link", 1, -1, 5, 10, np.nan, kind="transmission", border="EE-FI")
    row["available_mw"] = 350.0
    issues, target, idx = grid([0])
    values = outage_features(pd.DataFrame([row]), issues, target, idx)
    assert values["ic_fi_ee_unavailable_mw"].max() == 0
    reported = values["ic_fi_ee_reported_available_mw"]
    assert reported[4:9].tolist() == [350.0] * 5
    assert np.isnan(reported[:4]).all() and np.isnan(reported[9:]).all()


def test_two_notices_for_the_same_unit_do_not_double_count():
    rows = [event("notice1", 1, -1, 0, 48, 800), event("notice2", 1, -1, 0, 48, 1600)]
    for row in rows:
        row["resource"] = "OL3"
    issues, target, idx = grid([0])
    values = outage_features(pd.DataFrame(rows), issues, target, idx)
    assert values["nuclear_unavailable_mw"].max() == 1600
