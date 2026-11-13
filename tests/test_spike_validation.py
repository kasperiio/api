import pandas as pd

from scripts.validate_spike import promotion_checks, rolling_calibration


def test_future_actuals_cannot_change_earlier_calibration():
    rows = []
    for day in range(6):
        issue = pd.Timestamp("2025-01-01T12:00Z") + pd.Timedelta(days=day)
        for hour in range(1, 25):
            rows.append(dict(issued_at=issue, target_ts=issue + pd.Timedelta(hours=hour),
                             p10=10.0, p50=20.0, p90=30.0, actual=25.0))
    frame = pd.DataFrame(rows)
    cutoff = pd.Timestamp("2025-01-04T12:00Z")
    original = rolling_calibration(frame)
    frame.loc[frame.target_ts >= cutoff, "actual"] = 10000.0
    changed = rolling_calibration(frame)
    columns = ["p10_corrected", "p50_corrected", "p90_corrected"]
    pd.testing.assert_frame_equal(original.loc[original.issued_at <= cutoff, columns],
                                  changed.loc[changed.issued_at <= cutoff, columns])


def test_lower_median_error_cannot_override_bad_uncertainty_bands():
    baseline = dict(mae=20.0, last365_mae=21.0, spike_mae=60.0,
                    coverage=0.80, interval_score=100.0, mean_width=50.0)
    candidate = dict(baseline, mae=19.0, last365_mae=20.0, spike_mae=55.0,
                     coverage=0.70, interval_score=120.0)
    checks = promotion_checks(baseline, candidate)
    assert checks["lower_mae"] and checks["lower_spike_mae"]
    assert not checks["coverage"] and not checks["interval_score"]
    assert not all(checks.values())
