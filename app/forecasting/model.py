"""LightGBM quantile model for FI day-ahead prices.

Target transform: z = asinh(price / 10). It is ~linear around 0 (negative prices
down to the -500 floor are fine, unlike log) and ~logarithmic for spikes, so
heavy tails do not dominate the splits. Because quantiles are equivariant under
strictly monotone transforms, a quantile fitted in z-space maps back exactly to
the same quantile in EUR/MWh: p50 = 10 * sinh(q50(z)). (This would not hold for
a mean/L2 model, which is why we do not need any re-biasing.)

Persistence: a single JSON file holding the three booster model strings,
feature list, transform and params (no pickle -> no code execution on load).

Contributions: LightGBM ``pred_contrib`` (TreeSHAP) of the p50 booster, in the
transformed (asinh) space; columns are the features plus ``bias``, and they sum
to asinh(p50 / 10) before monotone sorting.
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
        self.num_boost_round = int(num_boost_round or DEFAULT_ROUNDS)
        self.feature_names: List[str] = []
        self.boosters: Dict[float, lgb.Booster] = {}
        self.meta: dict = {}

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
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Frame with columns p10, p50, p90 (EUR/MWh), same index as X, monotone p10<=p50<=p90."""
        if not self.boosters:
            raise RuntimeError("model is not fitted")
        data = self._matrix(X)
        nt = self._threads()
        z = np.column_stack([self.boosters[q].predict(data, num_threads=nt) for q in self.QUANTILES])
        z.sort(axis=1)  # enforce non-crossing quantiles
        return pd.DataFrame(inverse_transform(z), index=X.index, columns=list(self.COLUMNS))

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution (SHAP, via LightGBM pred_contrib) to p50, same index as X.

        Values are in the transformed asinh(price/10) space; the extra ``bias`` column is
        the expected value, and each row sums to the raw p50 prediction in that space."""
        if not self.boosters:
            raise RuntimeError("model is not fitted")
        contrib = self.boosters[0.5].predict(self._matrix(X), pred_contrib=True,
                                             num_threads=self._threads())
        return pd.DataFrame(contrib, index=X.index, columns=self.feature_names + ["bias"])

    def feature_importance(self, X: pd.DataFrame, top: Optional[int] = 20) -> pd.Series:
        """Mean |contribution| per feature on X, descending."""
        c = self.contributions(X).drop(columns="bias").abs().mean().sort_values(ascending=False)
        return c.head(top) if top else c

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        if not self.boosters:
            raise RuntimeError("model is not fitted")
        payload = {
            "format": "PriceModel",
            "format_version": self.FORMAT_VERSION,
            "transform": {"kind": "asinh", "scale": TRANSFORM_SCALE},
            "quantiles": list(self.QUANTILES),
            "feature_names": self.feature_names,
            "params": self.params,
            "num_boost_round": self.num_boost_round,
            "meta": self.meta,
            "boosters": {str(q): self.boosters[q].model_to_string() for q in self.QUANTILES},
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
        if payload.get("format") != "PriceModel":
            raise ValueError("not a PriceModel file: %s" % path)
        tr = payload.get("transform", {})
        if tr.get("kind") != "asinh" or float(tr.get("scale")) != TRANSFORM_SCALE:
            raise ValueError("unsupported target transform: %r" % tr)
        m = cls(params=payload.get("params"), num_boost_round=payload.get("num_boost_round"))
        m.feature_names = list(payload["feature_names"])
        m.meta = payload.get("meta", {})
        m.boosters = {
            float(q): lgb.Booster(model_str=s) for q, s in payload["boosters"].items()
        }
        return m
