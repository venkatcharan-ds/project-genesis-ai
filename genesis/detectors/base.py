"""Detector interface shared by every candidate solution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from genesis.data.synthetic_leak import Episode


def causal_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean using only past/current samples (deployable in real time)."""
    window = max(1, int(window))
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.empty_like(x, dtype=float)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - window)
    out[:] = (c[idx] - c[lo]) / (idx - lo)
    return out


class Detector(ABC):
    """A candidate solution. Must be causal: predict(t) may only use data <= t."""

    name: str = "detector"
    default_params: dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)}")
        self.params = {**self.default_params, **params}
        self.fitted = False

    @abstractmethod
    def fit(self, episodes: list[Episode]) -> "Detector":
        """Calibrate on training episodes. May use labels only to select leak-free samples."""

    @abstractmethod
    def predict(self, episode: Episode) -> np.ndarray:
        """Return a boolean alarm array with one entry per timestep."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}
