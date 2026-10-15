"""Multi-day site simulator (Milestone 3). SYNTHETIC DATA ONLY.

Milestone 1-2 days were independent: every day had a fresh meter bias, so a
detector could never learn "what yesterday looked like". Real meters are not
like that. Here each *site* is one instrumented pipe segment observed for
`days_per_site` consecutive days as ONE continuous time series, then cut into
days (Episodes) for evaluation. Across days:

* meter bias persists and drifts slowly (random walk); rare step changes model
  a meter being swapped or recalibrated
* meter delay (lag) persists; it can change mid-day ("meter delay error")
* leaks persist until repaired and can span several days; they can start at
  any minute, including late evening (so they are already active at midnight
  and during the next morning's calibration window)
* sensors drop out (NaN gaps); legitimate demand spikes occur

Each day's meta keeps the Milestone 1 keys (so all evaluators work) plus site
context: site_id, day_index, leak_event_id, preexisting_leak, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genesis.data.synthetic_leak import Episode, _diurnal

DAY = 1440


@dataclass
class SiteConfig:
    n_sites: int = 60
    days_per_site: int = 7
    base_demand: float = 100.0
    flow_noise_std: float = 1.0
    demand_ar_coef: float = 0.95
    demand_ar_std: float = 0.01
    meter_bias_std: float = 0.5           # initial per-meter bias, L/min
    drift_std_per_day: float = 0.05       # random-walk drift per meter per day, L/min
    meter_step_prob_per_day: float = 0.0  # P(meter swap / recalibration) per day
    meter_step_std: float = 1.5           # size of a meter step change, L/min
    max_lag_minutes: int = 2
    lag_change_prob_per_day: float = 0.0  # P(meter delay changes mid-day) per day
    leak_rate_per_day: float = 0.12       # expected new leaks per site-day
    leak_start_mode: str = "uniform"      # "uniform" | "night" (23:00-02:00, around calibration)
    preexisting_at_start_prob: float = 0.0  # P(site is already leaking when observation starts)
    leak_duration_hours: tuple[float, float] = (6.0, 72.0)
    burst_share: float = 0.5
    burst_fraction: tuple[float, float] = (0.04, 0.12)
    gradual_fraction: tuple[float, float] = (0.01, 0.05)
    gradual_ramp_minutes: tuple[int, int] = (120, 480)
    demand_spike_probability: float = 0.3  # per day
    demand_spike_fraction: tuple[float, float] = (0.15, 0.4)
    demand_spike_minutes: tuple[int, int] = (10, 60)
    dropout_prob_per_day: float = 0.0     # per sensor per day
    dropout_minutes: tuple[int, int] = (10, 180)
    pressure_nominal: float = 3.0
    pressure_flow_coef: float = 0.4
    pressure_leak_coef: float = 0.5
    pressure_noise_std: float = 0.01
    pressure_offset_std: float = 0.02         # per-site pressure-sensor offset, bar
    pressure_drift_std_per_day: float = 0.002  # pressure-sensor random-walk drift, bar/day
    calib_minutes: int = 120              # only used to tag days for analysis


def _leak_events(cfg: SiteConfig, total: int, rng: np.random.Generator) -> list[dict]:
    """Non-overlapping leak events; at least one day between repair and next start."""
    events: list[dict] = []

    def add(start: int, preexisting: bool) -> int:
        dur = int(rng.uniform(*cfg.leak_duration_hours) * 60)
        end = min(total, start + dur)
        events.append({"start": start, "end": end, "preexisting": preexisting})
        return end + DAY                   # earliest next start

    t = 0
    if rng.random() < cfg.preexisting_at_start_prob:
        # Started before observation began: already at full rate at minute 0.
        t = add(0, True)
    while True:
        if cfg.leak_start_mode == "night":
            # First night window (23:00-02:00) at or after t, then a random number of nights later.
            offset = int(rng.integers(-60, 120))
            day = int(np.ceil((t - offset) / DAY)) + int(rng.geometric(min(1.0, cfg.leak_rate_per_day))) - 1
            start = day * DAY + offset
        else:
            start = t + int(rng.geometric(cfg.leak_rate_per_day / DAY))
        if start >= total - 60:
            break
        t = add(int(start), False)
    for e in events:
        burst = rng.random() < cfg.burst_share
        e["type"] = "burst" if burst else "gradual"
        frac = cfg.burst_fraction if burst else cfg.gradual_fraction
        e["rate"] = float(rng.uniform(*frac)) * cfg.base_demand
        e["ramp"] = 1 if (burst or e["preexisting"]) else int(rng.integers(*cfg.gradual_ramp_minutes))
    return events


def generate_site(cfg: SiteConfig, site_id: int, rng: np.random.Generator, first_episode_id: int) -> list[Episode]:
    D, T = cfg.days_per_site, cfg.days_per_site * DAY
    t = np.arange(T) + rng.integers(0, DAY)

    ar = np.zeros(T)
    eps = rng.normal(0.0, cfg.demand_ar_std, T)
    for i in range(1, T):
        ar[i] = cfg.demand_ar_coef * ar[i - 1] + eps[i]
    demand = cfg.base_demand * _diurnal(t) * (1.0 + ar)
    spike_day = np.zeros(D, bool)
    for d in range(D):
        if rng.random() < cfg.demand_spike_probability:
            dur = int(rng.integers(*cfg.demand_spike_minutes))
            s = d * DAY + int(rng.integers(0, DAY - dur))
            demand[s:s + dur] += float(rng.uniform(*cfg.demand_spike_fraction)) * cfg.base_demand
            spike_day[d] = True

    leak = np.zeros(T)
    event_id = np.full(T, -1)
    events = _leak_events(cfg, T, rng)
    for k, e in enumerate(events):
        n = e["end"] - e["start"]
        leak[e["start"]:e["end"]] = e["rate"] * np.clip((np.arange(n) + 1) / e["ramp"], 0.0, 1.0)
        event_id[e["start"]:e["end"]] = k

    # Meter bias: per-minute random walk (slow drift) plus rare step changes.
    step = cfg.drift_std_per_day / np.sqrt(DAY)
    bias_in = rng.normal(0, cfg.meter_bias_std) + np.cumsum(rng.normal(0, step, T))
    bias_out = rng.normal(0, cfg.meter_bias_std) + np.cumsum(rng.normal(0, step, T))
    step_day = np.zeros(D, bool)
    for d in range(D):
        if rng.random() < cfg.meter_step_prob_per_day:
            bias_in[d * DAY + int(rng.integers(0, DAY)):] += rng.normal(0, cfg.meter_step_std)
            step_day[d] = True

    lag = np.full(T, int(rng.integers(0, cfg.max_lag_minutes + 1)))
    lag_change_day = np.zeros(D, bool)
    for d in range(D):
        if rng.random() < cfg.lag_change_prob_per_day:
            lag[d * DAY + int(rng.integers(0, DAY)):] = int(rng.integers(0, cfg.max_lag_minutes + 4))
            lag_change_day[d] = True
    demand_seen_out = demand[np.maximum(0, np.arange(T) - lag)]

    flow_in = demand + leak + bias_in + rng.normal(0, cfg.flow_noise_std, T)
    flow_out = demand_seen_out + bias_out + rng.normal(0, cfg.flow_noise_std, T)
    pressure = (cfg.pressure_nominal - cfg.pressure_flow_coef * ((demand + leak) / cfg.base_demand) ** 2
                - cfg.pressure_leak_coef * leak / cfg.base_demand + rng.normal(0, cfg.pressure_noise_std, T)
                + rng.normal(0, cfg.pressure_offset_std)
                + np.cumsum(rng.normal(0, cfg.pressure_drift_std_per_day / np.sqrt(DAY), T)))

    missing = np.zeros(T)
    for arr in (flow_in, flow_out, pressure):
        for d in range(D):
            if rng.random() < cfg.dropout_prob_per_day:
                dur = int(rng.integers(*cfg.dropout_minutes))
                s = d * DAY + int(rng.integers(0, DAY - dur))
                arr[s:s + dur] = np.nan
                missing[s:s + dur] = 1

    days = []
    for d in range(D):
        sl = slice(d * DAY, (d + 1) * DAY)
        lr, ev = leak[sl], event_id[sl]
        active = np.flatnonzero(lr > 0)
        meta = {"site_id": site_id, "day_index": d, "has_leak": bool(active.size),
                "leak_type": None, "leak_start": None, "leak_final_rate": 0.0,
                "leak_peak_rate": float(lr.max()), "demand_spike": bool(spike_day[d]),
                "meter_bias": float(np.mean(bias_in[sl] - bias_out[sl])),
                "lag_minutes": int(lag[sl][0]), "lag_change": bool(lag_change_day[d]),
                "meter_step": bool(step_day[d]), "missing_minutes": int(missing[sl].sum()),
                "preexisting_leak": False, "leak_in_calibration": False,
                "leak_event_id": None, "leak_event_start_abs": None}
        if active.size:
            e = events[int(ev[active[0]])]
            meta.update(leak_type=e["type"], leak_start=int(active[0]), leak_final_rate=e["rate"],
                        leak_event_id=f"{site_id}-{int(ev[active[0]])}", leak_event_start_abs=e["start"],
                        preexisting_leak=bool(active[0] == 0),
                        leak_in_calibration=bool(active[0] < cfg.calib_minutes))
        days.append(Episode(first_episode_id + d, flow_in[sl].copy(), flow_out[sl].copy(),
                            pressure[sl].copy(), lr.copy(), lr > 0, meta))
    return days


def generate_sites(cfg: SiteConfig, seed: int) -> list[Episode]:
    rng = np.random.default_rng(seed)
    out: list[Episode] = []
    for s in range(cfg.n_sites):
        out += generate_site(cfg, s, rng, first_episode_id=s * cfg.days_per_site)
    return out


def split_sites(episodes: list[Episode], fractions: dict[str, float], seed: int) -> dict[str, list[Episode]]:
    """Split by whole SITE so no site's days (shared meters, multi-day leaks) cross splits."""
    total = sum(fractions.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"split fractions must sum to 1, got {total}")
    sites = sorted({e.meta["site_id"] for e in episodes})
    order = [sites[i] for i in np.random.default_rng(seed).permutation(len(sites))]
    out, cursor = {}, 0
    names = list(fractions)
    for i, name in enumerate(names):
        count = len(sites) - cursor if i == len(names) - 1 else int(round(fractions[name] * len(sites)))
        chosen = set(order[cursor:cursor + count])
        out[name] = [e for e in episodes if e.meta["site_id"] in chosen]   # keeps day order
        cursor += count
    return out
