"""Causal signal corrections for the mass-balance residual (Milestone 2).

Two corrections, both estimated once per day from a calibration window at the
start of the episode (minutes [0, calib_minutes)) and then frozen:

1. Meter-delay (lag) compensation. The outlet meter sees demand `lag` minutes
   after the inlet meter, so on fast demand changes the raw residual
   flow_in(t) - flow_out(t) shows a transient that looks like a leak. We pick
   the lag L in [0, max_lag] that minimizes the variance of the aligned residual
   flow_in(t - L) - flow_out(t) inside the calibration window. Aligning by
   delaying the inlet (not advancing the outlet) keeps the correction causal.

2. Daily baseline recalibration. Meter calibration bias shows up as a constant
   residual offset. We subtract the mean aligned residual of the calibration
   window.

KEY ASSUMPTION (tested by the stress experiment, not assumed safe): the
calibration window is leak-free. A leak already present during calibration is
absorbed into the baseline and becomes invisible to every corrected detector.

Before the window closes there is not enough data to estimate the lag, so the
raw residual is used with an expanding-mean bias estimate. Experiments should
score only after `calib_minutes` (evaluation.score_from_minute) so that all
candidates are compared on the same samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from genesis.data.synthetic_leak import Episode


@dataclass(frozen=True)
class Calibration:
    lag: int
    bias: float


def estimate_lag(flow_in: np.ndarray, flow_out: np.ndarray, calib: int, max_lag: int) -> int:
    best_lag, best_var = 0, np.inf
    for lag in range(0, max_lag + 1):
        if calib - lag < 10:          # too few aligned samples to trust
            break
        var = float(np.var(flow_in[: calib - lag] - flow_out[lag:calib]))
        if var < best_var:
            best_lag, best_var = lag, var
    return best_lag


def calibrate(ep: Episode, calib_minutes: int, lag_comp: bool, recalibrate: bool, max_lag: int) -> Calibration:
    calib = min(calib_minutes, len(ep))
    lag = estimate_lag(ep.flow_in, ep.flow_out, calib, max_lag) if lag_comp else 0
    bias = float(np.mean(ep.flow_in[: calib - lag] - ep.flow_out[lag:calib])) if recalibrate else 0.0
    return Calibration(lag, bias)


def corrected_residual(ep: Episode, *, lag_comp: bool, recalibrate: bool,
                       calib_minutes: int, max_lag: int) -> np.ndarray:
    raw = ep.residual
    if not (lag_comp or recalibrate):
        return raw
    n, calib = len(ep), min(calib_minutes, len(ep))
    cal = calibrate(ep, calib, lag_comp, recalibrate, max_lag)

    out = np.empty(n, dtype=float)
    # Calibration window: raw residual, minus an expanding (causal) bias estimate.
    head = raw[:calib]
    out[:calib] = head - (np.cumsum(head) / np.arange(1, calib + 1) if recalibrate else 0.0)
    # After calibration: aligned residual minus the frozen daily bias.
    lag = cal.lag
    out[calib:] = ep.flow_in[calib - lag: n - lag] - ep.flow_out[calib:] - cal.bias
    return out
