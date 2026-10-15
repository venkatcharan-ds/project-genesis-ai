"""Milestone 3: leak-safe calibration with an independent raw-signal safety net.

Milestone 2's daily recalibration subtracts the mean residual of each morning's
calibration window. If a leak is already running in that window it becomes
part of the "baseline" and disappears. This detector keeps the Milestone 2 core
(meter-delay alignment + bias-corrected mass balance + threshold), and wraps it
in safeguards that can each be switched on/off for ablation:

carry_forward    Keep a TRUSTED bias/lag per site across days. Use it from
                 minute 0 (no blind calibration window) and update it only
                 from calibrations that were accepted (EWMA, `ewma_alpha`).
reject_abnormal  Compare today's calibration bias to its reference (trusted
                 history, or the training population prior on a cold start).
                 |z| > tau  =>  do NOT adopt it. A pre-existing leak shows up
                 exactly like this: an upward jump of the calibration bias.
cal_alarm        If the calibration is abnormally HIGH (leak-like), raise a
                 calibration alarm for the rest of the day instead of trusting
                 either baseline. Never silently absorb it.
raw_net          Independent raw mass-balance detector (no calibration at all,
                 the Milestone 1 method) OR-ed into the output. It cannot be
                 fooled by calibration, at the price of a coarser threshold.
robust_cal       Median (not mean) calibration, so demand spikes and delay
                 transients in the window do not skew it; windows with too much
                 missing data are marked LOW_QUALITY and not adopted.
missing_hold     Sensor gaps are treated as "no evidence" (NaN-aware smoothing)
                 instead of forward-filled values, and the fused alarm state is
                 held through gaps.

When the calibration is not accepted (abnormal or low quality),
`uncertain_fallback` chooses how the core runs for the rest of that day:
"carry" (trusted/prior baseline), or "raw_only" (core suppressed, only the raw
net and calibration alarm decide). `rebaseline_after_days` > 0 force-accepts a
level shift that persists that many days (handles meter swaps, but would
eventually absorb a leak that long; 0 = never, an operator must re-baseline).

With all safeguards off this reduces EXACTLY to Milestone 2's
mass_balance_threshold(lag_comp=True, recalibrate=True); a test enforces it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from genesis.data.synthetic_leak import Episode
from genesis.detectors.base import Detector, causal_moving_average, filled, ffill, require_persistence

DAY_STATUS = ("OK", "ABNORMAL_HIGH", "ABNORMAL_LOW", "LOW_QUALITY", "LEARNING_BLOCKED", "REBASELINED")


def nan_causal_ma(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean over the valid (non-NaN) samples in the window; NaN if none."""
    window = max(1, int(window))
    ok = ~np.isnan(x)
    c = np.cumsum(np.insert(np.where(ok, x, 0.0), 0, 0.0))
    n = np.cumsum(np.insert(ok.astype(float), 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(0, idx - window)
    cnt = n[idx] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (c[idx] - c[lo]) / np.maximum(cnt, 1), np.nan)


class LeakSafeDetector(Detector):
    name = "leak_safe"
    default_params: dict[str, Any] = {
        "window": 60, "k": 3.0,                       # core threshold (as Milestone 2)
        "carry_forward": True, "reject_abnormal": True, "cal_alarm": True,
        "raw_net": True, "robust_cal": True, "missing_hold": True,
        "tau_shift": 4.0,          # z threshold vs trusted history
        "tau_prior": 3.0,          # z threshold vs population prior (cold start)
        "ewma_alpha": 0.5,         # trusted-bias update weight for accepted days
        "raw_window": 60, "raw_k": 5.0,               # raw safety-net threshold
        "max_missing_frac": 0.2,   # calibration window quality
        "uncertain_fallback": "carry",                # "carry" | "raw_only"
        "rebaseline_after_days": 0,
        "block_learning_after_alarm": True,
        "pressure_check": True,    # corroborate calibration shifts with line pressure
        "tau_pressure": 4.0,       # z threshold for a pressure drop / normal pressure
    }

    # ------------------------------------------------------------------ helpers
    def _signals(self, ep: Episode) -> tuple[np.ndarray, np.ndarray]:
        if self.params["missing_hold"]:
            return ep.flow_in.astype(float), ep.flow_out.astype(float)
        f = filled(ep)
        return f.flow_in, f.flow_out

    def _calibration(self, fin: np.ndarray, fout: np.ndarray) -> tuple[int, float, float]:
        """Candidate (lag, bias, missing fraction) from today's calibration window."""
        C, max_lag = min(self.params["calib_minutes"], len(fin)), self.params["max_lag"]
        best_lag, best_var = 0, np.inf
        for lag in range(0, max_lag + 1):
            if C - lag < 10:
                break
            d = fin[: C - lag] - fout[lag:C]
            if np.sum(~np.isnan(d)) < 10:
                continue
            var = float(np.nanvar(d)) if np.isnan(d).any() else float(np.var(d))
            if var < best_var:
                best_lag, best_var = lag, var
        d = fin[: C - best_lag] - fout[best_lag:C]
        miss = float(np.mean(np.isnan(d))) if len(d) else 1.0
        if miss >= 1.0:
            return best_lag, float("nan"), 1.0
        agg = np.nanmedian if self.params["robust_cal"] else (np.nanmean if np.isnan(d).any() else np.mean)
        return best_lag, float(agg(d)), miss

    @staticmethod
    def _aligned(fin: np.ndarray, fout: np.ndarray, lag: int, tail: np.ndarray | None) -> np.ndarray:
        """fin(t - lag) - fout(t); the first `lag` samples use yesterday's tail if known."""
        if lag == 0:
            return fin - fout
        prev = tail[-lag:] if tail is not None and len(tail) >= lag else fin[:lag]
        return np.concatenate([prev, fin[: len(fin) - lag]]) - fout

    def _pressure_level(self, ep: Episode, fout: np.ndarray) -> float:
        """Median pressure residual (observed - expected from flow) in the calibration
        window; NaN if pressure is mostly missing. A leak lowers it; a flow-meter
        swap does not, which is what lets us tell the two apart."""
        C = min(self.params["calib_minutes"], len(ep))
        pr = ep.pressure[:C] - (self.p_coef_[0] + self.p_coef_[1] * fout[:C] ** 2)
        ok = ~np.isnan(pr)
        return float(np.median(pr[ok])) if ok.mean() >= 0.5 else float("nan")

    def _smooth(self, x: np.ndarray, w: int) -> np.ndarray:
        return nan_causal_ma(x, w) if self.params["missing_hold"] else causal_moving_average(x, w)

    def _day(self, ep: Episode, st: dict[str, Any]) -> dict[str, Any]:
        """Residual + calibration decision for one day, updating site state `st`."""
        p = self.params
        fin, fout = self._signals(ep)
        n, C = len(fin), min(p["calib_minutes"], len(fin))
        lag_c, b_c, miss = self._calibration(fin, fout)
        p_lvl = self._pressure_level(ep, fout) if p["pressure_check"] else float("nan")

        trusted = p["carry_forward"] and st["bias"] is not None
        ref, sig, tau = ((st["bias"], self.shift_sigma_, p["tau_shift"]) if trusted
                         else (self.prior_mu_, self.prior_sigma_, p["tau_prior"]))
        z = (b_c - ref) / sig if not np.isnan(b_c) else 0.0

        # Pressure evidence (None when the check is off or the sensor is missing).
        zp = None
        if p["pressure_check"] and not np.isnan(p_lvl):
            p_trusted = trusted and st["p_base"] is not None
            p_ref, p_sig = ((st["p_base"], self.p_shift_sigma_) if p_trusted
                            else (self.p_prior_mu_, self.p_prior_sigma_))
            zp = (p_lvl - p_ref) / p_sig
        # Asymmetric on purpose: absorbing a leak is the costly error, so "normal" requires
        # no meaningful drop (z >= -tau/2); the band between -tau and -tau/2 is ambiguous
        # and is treated as leak-like by the status logic below.
        pressure_drop = zp is not None and zp < -p["tau_pressure"]
        pressure_normal = zp is not None and -p["tau_pressure"] / 2 <= zp <= p["tau_pressure"]
        pressure_ambiguous = zp is not None and not pressure_normal and not pressure_drop

        # Detection of an abnormal window is always computed (for cal_alarm and
        # diagnostics); whether it is REJECTED is the reject_abnormal switch.
        abnormal_high, abnormal_low = bool(z > tau), bool(z < -tau)
        status = "OK"
        if np.isnan(b_c) or (p["robust_cal"] and miss > p["max_missing_frac"]):
            status = "LOW_QUALITY"
        elif (abnormal_high or abnormal_low) and pressure_normal:
            status = "METER_SHIFT"        # flow baseline moved but pressure did not: re-baseline
        elif abnormal_high or pressure_drop or (pressure_ambiguous and zp < 0 and z > tau / 2):
            status = "ABNORMAL_HIGH"      # leak-like (or unverifiable: pressure missing)
        elif abnormal_low:
            status = "ABNORMAL_LOW"
        elif trusted and p["block_learning_after_alarm"] and st["alarm_at_midnight"]:
            # An alarm was running at midnight: do not learn today's window, unless
            # pressure positively confirms there is no leak.
            status = "OK" if pressure_normal else "LEARNING_BLOCKED"
        # A level shift that persists for N days is accepted as the new normal (meter swap).
        if (status in ("ABNORMAL_HIGH", "ABNORMAL_LOW") and p["rebaseline_after_days"] > 0
                and st["suspect_streak"] >= p["rebaseline_after_days"]):
            status = "REBASELINED"
        leak_like = status == "ABNORMAL_HIGH"
        rejected = (status in ("LOW_QUALITY", "LEARNING_BLOCKED")
                    or (status in ("ABNORMAL_HIGH", "ABNORMAL_LOW") and (p["reject_abnormal"] or pressure_drop)))
        accepted = not rejected
        st["suspect_streak"] = 0 if accepted else st["suspect_streak"] + 1

        # Baseline used after the window.
        if accepted:
            jump = (not trusted) or status in ("REBASELINED", "METER_SHIFT")
            bias = b_c if jump else (1 - p["ewma_alpha"]) * st["bias"] + p["ewma_alpha"] * b_c
            lag = lag_c
        else:
            bias = st["bias"] if trusted else self.prior_mu_
            lag = st["lag"] if (trusted and st["lag"] is not None) else lag_c

        r = np.empty(n, dtype=float)
        if trusted:      # carry-forward: calibrated residual from minute 0
            r[:C] = self._aligned(fin, fout, st["lag"], st["tail"])[:C] - st["bias"]
        else:            # Milestone 2: raw residual minus expanding mean (effectively blind)
            head = fin[:C] - fout[:C]
            if p["missing_hold"]:
                ok = ~np.isnan(head)
                cnt = np.maximum(np.cumsum(ok), 1)
                r[:C] = head - np.cumsum(np.where(ok, head, 0.0)) / cnt
            else:
                r[:C] = head - np.cumsum(head) / np.arange(1, C + 1)
        r[C:] = self._aligned(fin, fout, lag, st["tail"])[C:] - bias
        defined = np.ones(n, bool) if trusted else (np.arange(n) >= C)

        if p["carry_forward"] and accepted:   # only trusted calibrations update the state
            st["bias"], st["lag"] = bias, lag
            if not np.isnan(p_lvl):
                st["p_base"] = p_lvl if st["p_base"] is None else (
                    (1 - p["ewma_alpha"]) * st["p_base"] + p["ewma_alpha"] * p_lvl)
        st["tail"] = fin[-(p["max_lag"] + 5):]
        return {"residual": r, "defined": defined, "status": status, "z": float(z), "b_cal": b_c,
                "leak_like": leak_like, "z_pressure": None if zp is None else float(zp),
                "bias_used": float(bias), "lag_used": int(lag), "accepted": accepted,
                "valid": ~(np.isnan(fin) | np.isnan(fout))}

    @staticmethod
    def _new_state() -> dict[str, Any]:
        return {"bias": None, "lag": None, "tail": None, "suspect_streak": 0, "p_base": None,
                "alarm_at_midnight": False, "last_alarm": False}

    @staticmethod
    def _by_site(episodes: list[Episode]) -> list[list[Episode]]:
        groups: dict[Any, list[Episode]] = {}
        for ep in episodes:
            groups.setdefault(ep.meta.get("site_id", ("ep", ep.episode_id)), []).append(ep)
        return [sorted(g, key=lambda e: e.meta.get("day_index", 0)) for g in groups.values()]

    # -------------------------------------------------------------------- fit
    def fit(self, episodes: list[Episode]) -> "LeakSafeDetector":
        return self._fit(episodes)        # handles gaps itself (no blanket forward-fill)

    def _fit(self, episodes: list[Episode]) -> "LeakSafeDetector":
        p, C = self.params, self.params["calib_minutes"]
        sites = self._by_site(episodes)

        # 0. Pressure model: pressure ~ c0 + c1 * flow_out^2 on leak-free, valid samples.
        q = np.concatenate([ep.flow_out[~ep.label] for ep in episodes])
        pr = np.concatenate([ep.pressure[~ep.label] for ep in episodes])
        ok = ~(np.isnan(q) | np.isnan(pr))
        self.p_coef_ = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), q[ok] ** 2]), pr[ok], rcond=None)[0]

        # 1. Calibration statistics on leak-free training days (labels select clean days only).
        b_all, shifts, p_all, p_shifts = [], [], [], []
        for days in sites:
            prev = p_prev = None
            for ep in days:
                fin, fout = self._signals(ep)
                _, b, miss = self._calibration(fin, fout)
                pl = self._pressure_level(ep, fout)
                clean = not ep.label.any() and not np.isnan(b)
                if clean:
                    b_all.append(b)
                    if prev is not None:
                        shifts.append(b - prev)
                    if not np.isnan(pl):
                        p_all.append(pl)
                        if p_prev is not None:
                            p_shifts.append(pl - p_prev)
                prev = b if clean else None
                p_prev = pl if (clean and not np.isnan(pl)) else None
        self.p_prior_mu_ = float(np.mean(p_all)) if p_all else 0.0
        self.p_prior_sigma_ = float(np.std(p_all)) if len(p_all) > 1 else 1.0
        self.p_shift_sigma_ = float(np.std(p_shifts)) if len(p_shifts) > 1 else self.p_prior_sigma_
        self.prior_mu_ = float(np.mean(b_all)) if b_all else 0.0
        self.prior_sigma_ = float(np.std(b_all)) if len(b_all) > 1 else 1.0
        # Single-day datasets have no consecutive days; fall back to the population spread.
        self.shift_sigma_ = float(np.std(shifts)) if len(shifts) > 1 else self.prior_sigma_

        # 2. Raw safety net: Milestone 1 calibration (population, no per-day correction).
        raw = [self._smooth(self._signals(ep)[0] - self._signals(ep)[1], p["raw_window"])[~ep.label]
               for ep in episodes]
        raw = np.concatenate(raw)
        raw = raw[~np.isnan(raw)]
        self.raw_mu_, self.raw_sigma_ = float(raw.mean()), float(raw.std()) or 1e-9

        # 3. Core threshold: run the tracker over training sites (no alarms yet, so no
        #    learning-block) and use smoothed residuals on leak-free, defined samples.
        core = []
        for days in sites:
            st = self._new_state()
            for ep in days:
                d = self._day(ep, st)
                s = self._smooth(d["residual"], p["window"])
                m = ~ep.label & d["defined"] & ~np.isnan(s)
                core.append(s[m])
        core = np.concatenate(core)
        self.mu_, self.sigma_ = float(core.mean()), float(core.std()) or 1e-9
        self.fitted = True
        return self

    # ---------------------------------------------------------------- predict
    def raw_alarms(self, episode: Episode) -> np.ndarray:    # pragma: no cover - not used
        return self.predict(episode)

    def predict(self, episode: Episode) -> np.ndarray:
        """Single day with no site history (cold start)."""
        return self.predict_sequence([episode])[0]

    def predict_sequence(self, days: list[Episode]) -> list[np.ndarray]:
        p = self.params
        st = self._new_state()
        out = []
        self.components_ = getattr(self, "components_", {})
        for ep in days:
            d = self._day(ep, st)
            n, C = len(ep), min(p["calib_minutes"], len(ep))
            s = self._smooth(d["residual"], p["window"])
            core = np.nan_to_num(s, nan=-np.inf) > self.mu_ + p["k"] * self.sigma_
            if not d["accepted"] and p["uncertain_fallback"] == "raw_only":
                core[C:] = False
            core = require_persistence(core, p["persistence"])

            raw = np.zeros(n, bool)
            if p["raw_net"]:
                fin, fout = self._signals(ep)
                rs = self._smooth(fin - fout, p["raw_window"])
                raw = require_persistence(np.nan_to_num(rs, nan=-np.inf)
                                          > self.raw_mu_ + p["raw_k"] * self.raw_sigma_, p["persistence"])

            cal = np.zeros(n, bool)
            if p["cal_alarm"] and d["leak_like"]:
                # Leak-like calibration: alarm from the end of the window for as long as
                # the residual stays above half of the abnormal shift (ends on repair).
                level = max(0.5 * (d["b_cal"] - d["bias_used"]), self.mu_ + self.sigma_)
                cal[C:] = np.nan_to_num(s[C:], nan=-np.inf) > level

            fused = core | raw | cal
            if p["missing_hold"]:
                # No evidence during a gap: hold the last decision (yesterday's at a day start).
                held = np.where(d["valid"], fused.astype(float), np.nan)
                fused = ffill(held, lead=float(st["last_alarm"])) > 0.5
            st["last_alarm"] = bool(fused[-1])
            st["alarm_at_midnight"] = bool(fused[-60:].any())
            self.components_[ep.episode_id] = {"core": core, "raw": raw, "cal": cal,
                                               "status": d["status"], "z": d["z"], "z_pressure": d["z_pressure"],
                                               "bias_used": d["bias_used"], "b_cal": d["b_cal"]}
            out.append(fused)
        return out
