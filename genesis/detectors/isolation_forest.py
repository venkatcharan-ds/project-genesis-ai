"""Approach C: Isolation Forest on engineered multi-sensor features.

Unsupervised ML baseline: learn what "normal" looks like from leak-free
training samples using both the flow residual and a pressure residual
(observed pressure minus a pressure-vs-flow curve fitted on training data).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from genesis.data.synthetic_leak import Episode
from genesis.detectors.base import Detector, causal_moving_average


class IsolationForestDetector(Detector):
    name = "isolation_forest"
    default_params = {"window": 15, "threshold_quantile": 0.01, "n_estimators": 100,
                      "max_train_samples": 20000, "random_state": 0}

    def _features(self, ep: Episode) -> np.ndarray:
        w = self.params["window"]
        res = causal_moving_average(ep.residual, w)
        p_expected = self.p_coef_[0] + self.p_coef_[1] * ep.flow_out ** 2
        p_res = causal_moving_average(ep.pressure - p_expected, w)
        return np.column_stack([res, p_res])

    def fit(self, episodes: list[Episode]) -> "IsolationForestDetector":
        # Pressure ~ c0 + c1 * flow_out^2, fitted on leak-free samples only.
        q = np.concatenate([ep.flow_out[~ep.label] for ep in episodes])
        p = np.concatenate([ep.pressure[~ep.label] for ep in episodes])
        A = np.column_stack([np.ones_like(q), q ** 2])
        self.p_coef_ = np.linalg.lstsq(A, p, rcond=None)[0]

        X = np.concatenate([self._features(ep)[~ep.label] for ep in episodes])
        rng = np.random.default_rng(self.params["random_state"])
        if len(X) > self.params["max_train_samples"]:
            X = X[rng.choice(len(X), self.params["max_train_samples"], replace=False)]
        self.model_ = IsolationForest(n_estimators=self.params["n_estimators"],
                                      random_state=self.params["random_state"]).fit(X)
        # Lower score = more anomalous. Alarm below the q-quantile of normal scores.
        self.threshold_ = float(np.quantile(self.model_.score_samples(X), self.params["threshold_quantile"]))
        self.fitted = True
        return self

    def predict(self, episode: Episode) -> np.ndarray:
        X = self._features(episode)
        # Leaks increase inflow: only one-sided anomalies (positive residual) count.
        return (self.model_.score_samples(X) < self.threshold_) & (X[:, 0] > 0)
