"""Approach B: one-sided CUSUM on the standardized mass-balance residual.

CUSUM accumulates small, persistent positive deviations, so it should catch
slow (gradual) leaks that a fixed threshold misses, at the risk of drifting
upward under constant meter bias.
"""

from __future__ import annotations

import numpy as np

from genesis.data.synthetic_leak import Episode
from genesis.detectors.base import Detector


class CusumDetector(Detector):
    name = "cusum"
    default_params = {"drift": 0.5, "h": 20.0}

    def fit(self, episodes: list[Episode]) -> "CusumDetector":
        normal = np.concatenate([ep.residual[~ep.label] for ep in episodes])
        self.mu_ = float(normal.mean())
        self.sigma_ = float(normal.std()) or 1e-9
        self.fitted = True
        return self

    def predict(self, episode: Episode) -> np.ndarray:
        z = (episode.residual - self.mu_) / self.sigma_
        drift, h = self.params["drift"], self.params["h"]
        s = 0.0
        alarm = np.zeros(len(z), dtype=bool)
        # Sequential by nature; a plain loop is fast enough at 1440 samples/episode.
        for i, zi in enumerate(z):
            s = max(0.0, s + zi - drift)
            alarm[i] = s > h
        return alarm
