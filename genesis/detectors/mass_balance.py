"""Approach A: smoothed mass-balance threshold.

Physical intuition: in a sealed segment, inflow == outflow. A leak creates a
persistent positive residual. Smooth the residual causally and alarm when it
exceeds mean + k * std of the leak-free training residual.
"""

from __future__ import annotations

import numpy as np

from genesis.data.synthetic_leak import Episode
from genesis.detectors.base import Detector, causal_moving_average


class MassBalanceThreshold(Detector):
    name = "mass_balance_threshold"
    default_params = {"window": 15, "k": 4.0}

    def _fit(self, episodes: list[Episode]) -> "MassBalanceThreshold":
        smoothed = [causal_moving_average(self.residual(ep), self.params["window"])[self.fit_mask(ep)]
                    for ep in episodes]
        normal = np.concatenate(smoothed)
        self.mu_ = float(normal.mean())
        self.sigma_ = float(normal.std()) or 1e-9
        self.fitted = True
        return self

    def raw_alarms(self, episode: Episode) -> np.ndarray:
        r = causal_moving_average(self.residual(episode), self.params["window"])
        return r > self.mu_ + self.params["k"] * self.sigma_
