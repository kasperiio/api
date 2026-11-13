"""Tests for the LightGBM quantile PriceModel."""

import numpy as np
import pandas as pd
import pytest

from app.forecasting import features as F
from app.forecasting.model import PriceModel, inverse_transform, transform
from tests.test_features import SYN_START, make_session


@pytest.fixture(scope="module")
def frame(tmp_path_factory):
    db = make_session(tmp_path_factory.mktemp("model"), days=120)
    X, y, _ = F.build_training_frame(db, SYN_START + pd.Timedelta(days=10),
                                     SYN_START + pd.Timedelta(days=120))
    db.close()
    return X, y


@pytest.fixture(scope="module")
def fitted(frame):
    X, y = frame
    split = X.index.get_level_values("target_ts") < SYN_START + pd.Timedelta(days=95)
    m = PriceModel(num_boost_round=150).fit(X[split], y[split])
    return m, X[~split], y[~split]


def test_transform_roundtrip():
    p = np.array([-500.0, -5.0, 0.0, 3.5, 80.0, 1500.0])
    np.testing.assert_allclose(inverse_transform(transform(p)), p, rtol=1e-12, atol=1e-9)


def test_predict_shape_and_monotone(fitted):
    m, X, y = fitted
    pred = m.predict(X)
    assert list(pred.columns) == ["p10", "p50", "p90"]
    assert pred.index.equals(X.index)
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
    assert np.isfinite(pred.to_numpy()).all()


def test_beats_naive_on_holdout(fitted):
    m, X, y = fitted
    pred = m.predict(X)
    mae = (pred["p50"] - y).abs().mean()
    naive = (X["fi_same_hour_lastweek"] - y).abs().mean()
    assert mae < naive
    inside = ((y >= pred["p10"]) & (y <= pred["p90"])).mean()
    assert 0.5 < inside <= 1.0


def test_quantile_crossing_is_fixed():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    y = pd.Series(10 * X["a"] + rng.normal(0, 30, 400))
    m = PriceModel(params={"min_data_in_leaf": 5}, num_boost_round=30).fit(X, y)
    # Make the p10 booster predict above p90 by swapping them
    m.boosters[0.1], m.boosters[0.9] = m.boosters[0.9], m.boosters[0.1]
    pred = m.predict(X)
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()


def test_missing_and_reordered_columns(fitted):
    m, X, _ = fitted
    base = m.predict(X)
    shuffled = X[list(reversed(X.columns))]
    pd.testing.assert_frame_equal(m.predict(shuffled), base)
    dropped = X.drop(columns=["wind_cf"])
    assert np.isfinite(m.predict(dropped).to_numpy()).all()


def test_contributions_sum_to_p50(fitted):
    m, X, _ = fitted
    sub = X.iloc[:200]
    c = m.contributions(sub)
    assert list(c.columns) == list(X.columns) + ["bias"]
    assert c.index.equals(sub.index)
    raw = m.boosters[0.5].predict(sub.to_numpy(dtype=float))
    np.testing.assert_allclose(c.sum(axis=1).to_numpy(), raw, rtol=1e-6, atol=1e-6)
    imp = m.feature_importance(sub, top=5)
    assert len(imp) == 5 and imp.is_monotonic_decreasing


def test_sample_weight(frame):
    X, y = frame
    w = pd.Series(np.linspace(0.1, 1.0, len(y)), index=y.index)
    m = PriceModel(num_boost_round=20).fit(X, y, sample_weight=w)
    assert m.predict(X.iloc[:10]).shape == (10, 3)


def test_save_load_roundtrip(fitted, tmp_path):
    m, X, _ = fitted
    path = str(tmp_path / "sub" / "v1.model")
    m.save(path)
    m2 = PriceModel.load(path)
    assert m2.feature_names == m.feature_names
    pd.testing.assert_frame_equal(m2.predict(X), m.predict(X))
    pd.testing.assert_frame_equal(m2.contributions(X.iloc[:20]), m.contributions(X.iloc[:20]))


def test_load_rejects_foreign_file(tmp_path):
    p = tmp_path / "x.model"
    p.write_text('{"format": "other"}')
    with pytest.raises(ValueError):
        PriceModel.load(str(p))
