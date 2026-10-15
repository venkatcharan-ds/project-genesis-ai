"""Detector interface shared by every candidate solution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from genesis.data.synthetic_leak import Episode
from genesis.preprocess import corrected_residual

# Options every detector accepts. Defaults reproduce Milestone 1 behaviour exactly.
COMMON_PARAMS: dict[str, Any] = {
    "lag_comp": False,       # align inlet/outlet meters by the estimated delay
    "recalibrate": False,    # subtract the daily meter-bias baseline
    "calib_minutes": 120,    # calibration window at the start of each day
    "max_lag": 5,            # largest meter delay searched (minutes)
    "persistence": 1,        # raise an alarm only after N consecutive raw alarms
}


def causal_moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean using only past/current samples (deployable in real time)."""
    window = max(1, int(window))
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.empty_like(x, dtype=float)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - window)
    out[:] = (c[idx] - c[lo]) / (idx - lo)
    return out


def require_persistence(alarm: np.ndarray, n: int) -> np.ndarray:
    """True at t only if alarm[t-n+1 .. t] are all True (causal debounce)."""
    n = int(n)
    if n <= 1:
        return alarm
    c = np.cumsum(np.insert(alarm.astype(int), 0, 0))
    idx = np.arange(1, len(alarm) + 1)
    run = c[idx] - c[np.maximum(0, idx - n)]
    return (run == n) & (idx >= n)


def ffill(x: np.ndarray, lead: np.ndarray | float = 0.0) -> np.ndarray:
    """Causal forward-fill of NaN gaps. NaNs before the first valid sample have no
    past to copy, so they take `lead` (a value or a same-time array)."""
    if not np.isnan(x).any():
        return x
    idx = np.where(np.isnan(x), -1, np.arange(len(x)))
    np.maximum.accumulate(idx, out=idx)
    out = x[np.maximum(idx, 0)].astype(float)
    head = idx < 0
    out[head] = (lead[head] if isinstance(lead, np.ndarray) else lead)
    return out


def filled(ep: Episode) -> Episode:
    """Episode with sensor gaps forward-filled: the naive missing-data policy of the
    Milestone 1-2 detectors. Returns `ep` itself when nothing is missing, so results
    on gap-free data are bit-identical to earlier milestones. A flow meter missing at
    the start of the day mirrors the other meter (zero imbalance = no evidence)."""
    if not (np.isnan(ep.flow_in).any() or np.isnan(ep.flow_out).any() or np.isnan(ep.pressure).any()):
        return ep
    fin = ffill(ep.flow_in, lead=np.nan_to_num(ep.flow_out))
    fout = ffill(ep.flow_out, lead=fin)
    return Episode(ep.episode_id, fin, fout, ffill(ep.pressure), ep.leak_rate, ep.label, ep.meta)


class Detector(ABC):
    """A candidate solution. Must be causal: predict(t) may only use data <= t."""

    name: str = "detector"
    default_params: dict[str, Any] = {}

    def __init__(self, **params: Any) -> None:
        defaults = {**COMMON_PARAMS, **self.default_params}
        unknown = set(params) - set(defaults)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)}")
        self.params = {**defaults, **params}
        self.fitted = False

    # -- shared signal handling ------------------------------------------------
    @property
    def corrected(self) -> bool:
        return bool(self.params["lag_comp"] or self.params["recalibrate"])

    def residual(self, ep: Episode) -> np.ndarray:
        p = self.params
        return corrected_residual(ep, lag_comp=p["lag_comp"], recalibrate=p["recalibrate"],
                                  calib_minutes=p["calib_minutes"], max_lag=p["max_lag"])

    def fit_mask(self, ep: Episode) -> np.ndarray:
        """Samples used for calibration: leak-free, and past the calibration window
        when corrections are on (the window itself is transient by construction)."""
        mask = ~ep.label
        if self.corrected:
            mask = mask & (np.arange(len(ep)) >= self.params["calib_minutes"])
        return mask

    # -- interface ---------------------------------------------------------------
    def fit(self, episodes: list[Episode]) -> "Detector":
        """Calibrate on training episodes. May use labels only to select leak-free samples."""
        return self._fit([filled(ep) for ep in episodes])

    @abstractmethod
    def _fit(self, episodes: list[Episode]) -> "Detector":
        """Fit on gap-filled episodes."""

    @abstractmethod
    def raw_alarms(self, episode: Episode) -> np.ndarray:
        """Per-timestep alarm decision before persistence filtering (gap-filled input)."""

    def predict(self, episode: Episode) -> np.ndarray:
        """Return a boolean alarm array with one entry per timestep."""
        return require_persistence(self.raw_alarms(filled(episode)), self.params["persistence"])

    def predict_sequence(self, days: list[Episode]) -> list[np.ndarray]:
        """Predict consecutive days of ONE site, in order. Stateless detectors treat
        each day independently; stateful ones (leak_safe) carry state across days."""
        return [self.predict(d) for d in days]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}
