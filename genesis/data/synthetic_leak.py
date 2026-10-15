"""Synthetic water-distribution sensor data with injected leaks.

SYNTHETIC DATA ONLY. This simulator is a simplified model of a single pipe
segment instrumented with an inlet flow meter, an outlet flow meter and a
pressure sensor. Results obtained on it say nothing about physical-world
performance until validated on real measurements.

Model (1 sample per minute):
    demand(t)    = base * diurnal(t) * (1 + AR(1) noise) [+ demand spike]
    flow_out(t)  = demand(t - lag) + meter_bias_out + noise      (consumer side)
    flow_in(t)   = demand(t) + leak(t) + meter_bias_in + noise   (supply side)
    pressure(t)  = p_nom - a * (Q_in / base)^2 - b * leak / base + noise

Confounders deliberately included so that naive detectors fail sometimes:
    * per-episode meter calibration bias (constant residual offset)
    * transport lag between meters (transient residual on demand changes)
    * legitimate demand spikes (e.g. hydrant use) seen by both meters
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class GeneratorConfig:
    n_episodes: int = 120
    episode_minutes: int = 1440
    leak_probability: float = 0.5
    base_demand: float = 100.0          # L/min
    flow_noise_std: float = 1.0         # L/min, per meter, per sample
    demand_ar_coef: float = 0.95
    demand_ar_std: float = 0.01         # relative demand fluctuation innovation
    meter_bias_std: float = 0.5         # L/min, per meter, per episode
    max_lag_minutes: int = 2
    burst_fraction: tuple[float, float] = (0.04, 0.12)    # of base demand
    gradual_fraction: tuple[float, float] = (0.01, 0.05)  # final rate
    gradual_ramp_minutes: tuple[int, int] = (120, 480)
    burst_share: float = 0.5            # P(burst | leak)
    warmup_minutes: int = 120           # leaks never start before this
    demand_spike_probability: float = 0.3
    demand_spike_fraction: tuple[float, float] = (0.15, 0.4)
    demand_spike_minutes: tuple[int, int] = (10, 60)
    pressure_nominal: float = 3.0       # bar
    pressure_flow_coef: float = 0.4
    pressure_leak_coef: float = 0.5
    pressure_noise_std: float = 0.01


@dataclass
class Episode:
    episode_id: int
    flow_in: np.ndarray
    flow_out: np.ndarray
    pressure: np.ndarray
    leak_rate: np.ndarray          # ground truth, L/min (never visible to detectors)
    label: np.ndarray              # bool, True where a leak is active
    meta: dict = field(default_factory=dict)

    @property
    def residual(self) -> np.ndarray:
        """Mass-balance residual: water entering minus water leaving."""
        return self.flow_in - self.flow_out

    def __len__(self) -> int:
        return len(self.label)


def _diurnal(t: np.ndarray) -> np.ndarray:
    # Morning and evening peaks, night minimum; mean ~1.
    day = 2 * np.pi * t / 1440.0
    return 1.0 + 0.30 * np.sin(day - np.pi / 2) + 0.10 * np.sin(2 * day - np.pi / 3)


def generate_episode(cfg: GeneratorConfig, episode_id: int, rng: np.random.Generator) -> Episode:
    n = cfg.episode_minutes
    # Random phase offset so episodes do not all start at midnight.
    t = np.arange(n) + rng.integers(0, 1440)

    ar = np.zeros(n)
    eps = rng.normal(0.0, cfg.demand_ar_std, n)
    for i in range(1, n):
        ar[i] = cfg.demand_ar_coef * ar[i - 1] + eps[i]
    demand = cfg.base_demand * _diurnal(t) * (1.0 + ar)

    meta: dict = {"has_leak": False, "leak_type": None, "leak_start": None,
                  "leak_final_rate": 0.0, "demand_spike": False}

    if rng.random() < cfg.demand_spike_probability:
        dur = int(rng.integers(*cfg.demand_spike_minutes))
        start = int(rng.integers(0, n - dur))
        frac = float(rng.uniform(*cfg.demand_spike_fraction))
        demand[start:start + dur] += frac * cfg.base_demand
        meta.update(demand_spike=True, spike_start=start, spike_minutes=dur)

    leak = np.zeros(n)
    if rng.random() < cfg.leak_probability:
        start = int(rng.integers(cfg.warmup_minutes, n - 60))
        if rng.random() < cfg.burst_share:
            rate = float(rng.uniform(*cfg.burst_fraction)) * cfg.base_demand
            leak[start:] = rate
            kind = "burst"
        else:
            rate = float(rng.uniform(*cfg.gradual_fraction)) * cfg.base_demand
            ramp = int(rng.integers(*cfg.gradual_ramp_minutes))
            k = np.arange(n - start)
            leak[start:] = rate * np.clip((k + 1) / ramp, 0.0, 1.0)
            kind = "gradual"
        meta.update(has_leak=True, leak_type=kind, leak_start=start, leak_final_rate=rate)

    lag = int(rng.integers(0, cfg.max_lag_minutes + 1))
    demand_lagged = np.concatenate([np.full(lag, demand[0]), demand[: n - lag]]) if lag else demand
    bias_in = float(rng.normal(0.0, cfg.meter_bias_std))
    bias_out = float(rng.normal(0.0, cfg.meter_bias_std))
    meta.update(lag_minutes=lag, meter_bias=bias_in - bias_out)

    flow_in = demand + leak + bias_in + rng.normal(0.0, cfg.flow_noise_std, n)
    flow_out = demand_lagged + bias_out + rng.normal(0.0, cfg.flow_noise_std, n)
    pressure = (cfg.pressure_nominal
                - cfg.pressure_flow_coef * ((demand + leak) / cfg.base_demand) ** 2
                - cfg.pressure_leak_coef * leak / cfg.base_demand
                + rng.normal(0.0, cfg.pressure_noise_std, n))

    return Episode(episode_id, flow_in, flow_out, pressure, leak, leak > 0, meta)


def generate_dataset(cfg: GeneratorConfig, seed: int) -> list[Episode]:
    rng = np.random.default_rng(seed)
    return [generate_episode(cfg, i, rng) for i in range(cfg.n_episodes)]


def split_episodes(episodes: list[Episode], fractions: dict[str, float], seed: int) -> dict[str, list[Episode]]:
    """Split by whole episode (never by timestep) so no temporal leakage across splits."""
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")
    order = np.random.default_rng(seed).permutation(len(episodes))
    out, cursor = {}, 0
    names = list(fractions)
    for i, name in enumerate(names):
        count = len(episodes) - cursor if i == len(names) - 1 else int(round(fractions[name] * len(episodes)))
        out[name] = [episodes[j] for j in order[cursor:cursor + count]]
        cursor += count
    return out


def dataset_fingerprint(episodes: list[Episode]) -> str:
    """SHA-256 over all sensor + label arrays, for reproducibility evidence."""
    h = hashlib.sha256()
    for ep in episodes:
        for arr in (ep.flow_in, ep.flow_out, ep.pressure, ep.label):
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def config_to_dict(cfg: GeneratorConfig) -> dict:
    return asdict(cfg)
