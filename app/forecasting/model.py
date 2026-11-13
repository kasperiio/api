"""LightGBM quantile model for FI day-ahead prices.

Target transform: z = asinh(price / 10). It is ~linear around 0 (negative prices
down to the -500 floor are fine, unlike log) and ~logarithmic for spikes, so
heavy tails do not dominate the splits. Because quantiles are equivariant under
strictly monotone transforms, a quantile fitted in z-space maps back exactly to
the same quantile in EUR/MWh: p50 = 10 * sinh(q50(z)). (This would not hold for
a mean/L2 model, which is why we do not need any re-biasing.)

Persistence: a single JSON file holding the three booster model strings,
feature list, transform and params (no pickle -> no code execution on load).

Contributions: LightGBM ``pred_contrib`` (TreeSHAP) of the base p50 booster, in the
transformed (asinh) space; columns are the features plus ``bias``, and they sum
to asinh(p50 / 10) before monotone sorting. The optional spike mixture appends
an explicitly named ``spike_adjustment`` so the explanation sums to its final
median; that adjustment is not a feature SHAP value.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

TRANSFORM_SCALE = 10.0
# Total threads for one fit; the three quantile boosters train concurrently with
# a third each (LightGBM per-iteration overhead dominates at these data sizes,
# so few-thread boosters in parallel beat one many-thread booster).
DEFAULT_THREADS = max(1, min(12, os.cpu_count() or 1))

DEFAULT_PARAMS = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 48,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
}
DEFAULT_ROUNDS = 700
SPIKE_THRESHOLD = 100.0
MIN_SPIKE_CLASS_ROWS = 200


def transform(price) -> np.ndarray:
    return np.arcsinh(np.asarray(price, dtype=float) / TRANSFORM_SCALE)


def inverse_transform(z) -> np.ndarray:
    return TRANSFORM_SCALE * np.sinh(np.asarray(z, dtype=float))


class PriceModel:
    QUANTILES = (0.1, 0.5, 0.9)
    COLUMNS = ("p10", "p50", "p90")
    FORMAT_VERSION = 1

    def __init__(self, params: Optional[dict] = None, num_boost_round: Optional[int] = None):
        self.params = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.spike_mixture = bool(self.params.pop("spike_mixture", False))
        self.num_boost_round = int(num_boost_round or DEFAULT_ROUNDS)
        self.feature_names: List[str] = []
        self.boosters: Dict[float, lgb.Booster] = {}
        self.meta: dict = {}
        self.spike_boosters: Dict[str, lgb.Booster] = {}

    # ------------------------------------------------------------------
    def _threads(self) -> int:
        """Threads per booster for training and predict calls: num_threads / 3.

        Always passed explicitly: LightGBM's predict default is all cores, and those
        spinning OpenMP teams starve other models trained/used from other threads."""
        total = int(self.params.get("num_threads") or 0) or DEFAULT_THREADS
        return max(1, total // len(self.QUANTILES))

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Align columns to the training feature list (missing -> NaN)."""
        return X.reindex(columns=self.feature_names).to_numpy(dtype=np.float64)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[pd.Series] = None) -> "PriceModel":
        y_arr = np.asarray(y, dtype=float)
        mask = np.isfinite(y_arr)
        w = None
        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=float)
            mask &= np.isfinite(w) & (w > 0)
        if not mask.any():
            raise ValueError("no training rows with a finite target")
        self.feature_names = list(X.columns)
        data = self._matrix(X)[mask]
        z = transform(y_arr[mask])
        weight = None if w is None else w[mask]
        per_booster = self._threads()

        def train(q: float) -> lgb.Booster:
            ds = lgb.Dataset(  # one Dataset per booster: safe for concurrent training
                data, label=z, weight=weight, feature_name=self.feature_names,
                params={"max_bin": self.params.get("max_bin", 255), "verbose": -1},
            )
            params = dict(self.params, alpha=q, num_threads=per_booster)
            return lgb.train(params, ds, num_boost_round=self.num_boost_round)

        with ThreadPoolExecutor(max_workers=len(self.QUANTILES)) as pool:
            boosters = list(pool.map(train, self.QUANTILES))
        self.boosters = dict(zip(self.QUANTILES, boosters))
        self.meta = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "n_rows": int(mask.sum()),
        }
        self.spike_boosters = {}
        if self.spike_mixture:
            self.fit_spike(X, y, sample_weight=sample_weight)
        return self

    def fit_spike(self, X: pd.DataFrame, y: pd.Series,
                  sample_weight: Optional[pd.Series] = None) -> "PriceModel":
        """Fit a spike classifier and conditional medians on the base model's rows.

        The weighted conditional medians are an empirical point predictor, not
        a mathematical mixture quantile. The base p10/p90 bounds are retained,
        expanded to contain the new center, and calibrated by the feedback loop.
        """
        if not self.boosters:
            raise RuntimeError("fit the base quantile model before fitting spike experts")
        self.spike_mixture = True
        self.spike_boosters = {}
        values = np.asarray(y, dtype=float)
        valid = np.isfinite(values)
        weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weight is not None:
            valid &= np.isfinite(weight) & (weight > 0)
        high = values > SPIKE_THRESHOLD
        if min(np.sum(valid & high), np.sum(valid & ~high)) < MIN_SPIKE_CLASS_ROWS:
            self.meta["spike_status"] = "insufficient_class_rows"
            return self
        data = self._matrix(X)

        def train(kind):
            mask = valid if kind == "classifier" else valid & (high if kind == "high" else ~high)
            label = high.astype(float) if kind == "classifier" else transform(values)
            params = dict(self.params, num_threads=self._threads())
            if kind == "classifier":
                params.update(objective="binary")
                params.pop("alpha", None)
            else:
                params.update(objective="quantile", alpha=0.5)
            ds = lgb.Dataset(data[mask], label=label[mask],
                             weight=None if weight is None else weight[mask],
                             feature_name=self.feature_names)
            return lgb.train(params, ds, num_boost_round=(min(300, self.num_boost_round)
                                                        if kind == "classifier" else self.num_boost_round))

        names = ("classifier", "low", "high")
        with ThreadPoolExecutor(max_workers=3) as pool:
            self.spike_boosters = dict(zip(names, pool.map(train, names)))
        self.meta["spike_status"] = "fitted"
        self.meta["spike_threshold_eur_mwh"] = SPIKE_THRESHOLD
        return self

    @property
    def explanation_method(self) -> str:
        return "base_model_shap_plus_spike_adjustment" if self.spike_boosters else "base_model_shap"

    def predict(self, X: pd.DataFrame, use_spike: bool = True) -> pd.DataFrame:
        """Frame with columns p10, p50, p90 (EUR/MWh), same index as X, monotone p10<=p50<=p90."""
        if not self.boosters:
            raise RuntimeError("model is not fitted")
        data = self._matrix(X)
        nt = self._threads()
        z = np.column_stack([self.boosters[q].predict(data, num_threads=nt) for q in self.QUANTILES])
        z.sort(axis=1)  # enforce non-crossing quantiles
        pred = inverse_transform(z)
        if use_spike and self.spike_boosters:
            p = self.spike_boosters["classifier"].predict(data, num_threads=nt)
            low = inverse_transform(self.spike_boosters["low"].predict(data, num_threads=nt))
            high = inverse_transform(self.spike_boosters["high"].predict(data, num_threads=nt))
            pred[:, 1] = (1 - p) * low + p * high
            pred[:, 0] = np.minimum(pred[:, 0], pred[:, 1])
            pred[:, 2] = np.maximum(pred[:, 2], pred[:, 1])
        return pd.DataFrame(pred, index=X.index, columns=list(self.COLUMNS))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution (SHAP, via LightGBM pred_contrib) to p50, same index as X.

        Values are in the transformed asinh(price/10) space; the extra ``bias`` column is
        the expected value. Spike models append an aggregate spike_adjustment,
        rather than claiming base-model SHAP explains the full mixture."""
        if not self.boosters:
            raise RuntimeError("model is not fitted")
        contrib = self.boosters[0.5].predict(self._matrix(X), pred_contrib=True,
                                             num_threads=self._threads())
        out = pd.DataFrame(contrib, index=X.index, columns=self.feature_names + ["bias"])
        if self.spike_boosters:
            out["spike_adjustment"] = transform(self.predict(X)["p50"]) - out.sum(axis=1)
        return out

    def feature_importance(self, X: pd.DataFrame, top: Optional[int] = 20) -> pd.Series:
        """Mean |contribution| per feature on X, descending."""
        c = self.contributions(X)[self.feature_names].abs().mean().sort_values(ascending=False)
        return c.head(top) if top else c

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        if not self.boosters:
            raise RuntimeError("model is not fitted")
        payload = {
            "format": "PriceModelSpikeMixture" if self.spike_boosters else "PriceModel",
            "format_version": 2 if self.spike_boosters else self.FORMAT_VERSION,
            "transform": {"kind": "asinh", "scale": TRANSFORM_SCALE},
            "quantiles": list(self.QUANTILES),
            "feature_names": self.feature_names,
            "params": dict(self.params, spike_mixture=self.spike_mixture),
            "num_boost_round": self.num_boost_round,
            "meta": self.meta,
            "boosters": {str(q): self.boosters[q].model_to_string() for q in self.QUANTILES},
            "spike_boosters": {name: b.model_to_string() for name, b in self.spike_boosters.items()},
        }
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "PriceModel":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("format") not in ("PriceModel", "PriceModelSpikeMixture"):
            raise ValueError("not a PriceModel file: %s" % path)
        if payload.get("format_version", 1) not in (1, 2):
            raise ValueError("unsupported PriceModel format version")
        tr = payload.get("transform", {})
        if tr.get("kind") != "asinh" or float(tr.get("scale")) != TRANSFORM_SCALE:
            raise ValueError("unsupported target transform: %r" % tr)
        m = cls(params=payload.get("params"), num_boost_round=payload.get("num_boost_round"))
        m.feature_names = list(payload["feature_names"])
        m.meta = payload.get("meta", {})
        m.boosters = {
            float(q): lgb.Booster(model_str=s) for q, s in payload["boosters"].items()
        }
        saved_spike = payload.get("spike_boosters", {})
        if payload["format"] == "PriceModelSpikeMixture" and not saved_spike:
            raise ValueError("missing spike experts")
        if saved_spike and set(saved_spike) != {"classifier", "low", "high"}:
            raise ValueError("incomplete spike model")
        m.spike_boosters = {name: lgb.Booster(model_str=s) for name, s in saved_spike.items()}
        return m
