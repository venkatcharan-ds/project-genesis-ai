# GENESIS Milestone 3: Leak-Safe Calibration and Robust Detection

> **SYNTHETIC EXPERIMENTS: NOT PHYSICAL-WORLD VALIDATION.** Every number below comes from the
> simulators in `genesis/data/`. The results describe how the algorithms behave on simplified models
> of a pipe segment. They do not show how these detectors would perform on real pipes, meters or
> pressure sensors.

## 1. The failure we set out to fix (kept, not hidden)

Milestone 2's daily recalibration subtracts the mean residual of each morning's 2-hour calibration
window. If a leak is already running in that window, it becomes part of the "baseline" and disappears.
Milestone 2's stress test B measured this: leak-day recall fell from 1.00 to 0.47 for the threshold detector.

- That run is still in the repository and **still reproduces**: `check-repro` on
  `20261015T051226Z_water_leak_m2_stress_preexisting_9b927739` → REPRODUCED.
- The failure is pinned as a regression test:
  `tests/test_milestone3.py::test_regression_m2_absorbs_leak_running_at_midnight`.

On the realistic multi-day data built for this milestone, the failure is not a corner case. Leaks last
6–72 h, so in the realistic runs **52–60% of leak days start with the leak already running**. Across all
multi-day scenarios, Milestone 2 catches only 6–44% of leak days.

## 2. What was built

### Multi-day site simulator (`genesis/data/sequences.py`)
Each *site* is one continuous multi-day time series, cut into days. Leak-free behaviour per site:
- the meter bias persists and drifts slowly, with rare step changes (meter swap or recalibration)
- the meter delay persists and can change mid-day (meter delay error)
- legitimate demand spikes occur
- sensors drop out (NaN gaps)
- the pressure sensor has its own per-site offset and slow drift

Leaks are scheduled per site:
- they persist across days until repaired
- they can start at any minute, or only at night (23:00–02:00) for stress tests
- for cold-start tests, a leak can already be running when observation begins

Splits are by **whole site**, and confidence intervals use a **site-clustered bootstrap**.

### `leak_safe` detector (`genesis/detectors/leak_safe.py`)
It keeps the Milestone 2 core (delay alignment, bias-corrected mass balance, threshold) and wraps it in
safeguards. Each one has its own switch for ablation.

| Safeguard | What it does | Requirement covered |
|---|---|---|
| `carry_forward` | Keeps a **trusted** bias and delay per site across days, used from minute 0 (no blind window). It is updated only by accepted calibrations. | Carry forward trusted bias |
| `reject_abnormal` | A calibration bias far from the trusted value (or from the population prior on a cold start) is **not adopted**. | Detect abnormal calibration windows |
| `pressure_check` | Pressure corroboration. If flow shifted **and** pressure dropped, it's a leak: reject and alarm. If flow shifted but pressure is normal, it's a **meter shift**: re-baseline. Ambiguous counts as leak-like. | Drift, meter swaps |
| `cal_alarm` | A leak-like calibration raises an alarm for as long as the abnormal level persists. Calibration never silently erases it. | No silent erasure |
| `raw_net` | An **independent raw mass-balance detector** (the Milestone 1 method, no calibration), OR-ed in. | Raw-signal safety net |
| `robust_cal` | Median (not mean) calibration. Windows with more than 20% missing data are LOW_QUALITY and not adopted. | Demand spikes, missing data |
| `missing_hold` | Sensor gaps count as "no evidence" (NaN-aware smoothing), and the alarm state is held through gaps. | Missing sensors |
| `block_learning_after_alarm` | No learning from a window that follows an active alarm, unless pressure confirms there is no leak. | Leaks spanning midnight |
| `uncertain_fallback`, `raw_k_uncertain`, `rebaseline_after_days` | Configurable safe fallback while the calibration state is uncertain: carry forward or raw-only, a more sensitive raw net, and optional forced re-baselining. | Safe fallback |

Each day ends in a recorded calibration status: `OK`, `ABNORMAL_HIGH`, `ABNORMAL_LOW`, `METER_SHIFT`,
`LOW_QUALITY`, `LEARNING_BLOCKED` or `REBASELINED`. Per-component alarms are kept for attribution.
**With every safeguard off, `leak_safe` reproduces the Milestone 2 detector exactly**: same fitted
thresholds and identical alarms, enforced by `test_all_safeguards_off_equals_milestone2`.

## 3. How it was tested, and how the design changed (full history)

All runs are in `experiments/runs/` and the ledger. Each confirmatory suite used **fresh seeds**, and
its hypotheses were **committed to git before those seeds were generated** (commits `f2488a6`,
`2dc990e`, `09d617e`). Tuning used validation sites only. Test sites were scored once.

| Version | Change | Suite (seeds) | Verdicts | What it revealed |
|---|---|---|---|---|
| dev | first design; pressure check and level-based calibration alarm added after diagnosing **validation** false alarms (a meter swap caused a multi-day alarm lock-up) | 301 (development only) | n/a | M2 collapses on multi-day data; first design over the false-alarm budget |
| v1 | as built | 302–305, replay 45 | 16 supported · 8 inconclusive · 0 contradicted | **Cold start:** M3 caught 15.6 points fewer leak days than raw M1 (C2) |
| v2 | opt-in `meter_shift_requires_site_pressure` and `raw_k_uncertain`, after diagnosing the cold-start gap on **validation** data | 312–315, replay | 20 · 5 · **1 contradicted (R8)** | Cold start fixed, but false alarms rose. Cause: the **tuning objective never counted alarms**, so it picked flapping configurations, and the evaluation budget ran out |
| **v3** | no detector change; objective penalizes **false alarms per leak-free day**; budget 60; **all** candidates (incl. M1, M2) re-tuned | **322–325**, replay | **21 · 5 · 0** | final; numbers below |

Correction: the commit message of `c496eaa` says the v1 suite had "18 of 24 supported". The
recorded results show **16 supported, 8 inconclusive**. The recorded results are authoritative; git
history was not rewritten.

Multiple-comparison caveat: across the three confirmatory suites we evaluated 76 hypotheses. Verdicts use
unadjusted 95% intervals, so expect a few spurious results at this scale. The v3 conclusions below are
the ones that also held, in direction, in the v1 and v2 suites.

## 4. Final results (v3, held-out test sites; 32 sites × 7 days per scenario)

`m3_full` = all safeguards + v2 options. `m3_v1` = the same safeguards without the v2 options.
Brackets are 95% site-clustered bootstrap CIs. Runtime is about 0.4 µs per sample for M3 vs 0.04 for M1.

### Realistic mix — `20261015T191949Z_water_leak_m3v3_confirm_b2a287fc`
| Detector | Precision | Recall | F1 | Leak-day recall | Leak-event recall | False alarms / leak-free day | Mean event delay (min) | Missed events |
|---|---|---|---|---|---|---|---|---|
| M1 raw | 0.970 | 0.549 | 0.701 [0.47, 0.86] | 0.721 | 0.824 | 0.028 [0.006, 0.051] | 153 | 3 |
| M2 recalibration | 0.933 | 0.224 | 0.362 [0.26, 0.47] | 0.395 | 0.941 | 0.033 [0.006, 0.066] | 194 | 1 |
| Raw net alone | 0.967 | 0.604 | 0.744 | 0.791 | 0.941 | 0.033 | 146 | 1 |
| M3 v1 | 0.966 | 0.937 | 0.951 | 1.000 | 1.000 | 0.072 [0.012, 0.155] | 84 | 0 |
| **M3 full** | 0.968 | 0.923 | **0.945 [0.91, 0.97]** | **1.000** | **1.000** | 0.044 [0.011, 0.087] | 94 | **0** |

### Stress: leaks active at midnight or starting during calibration — `…m3v3_stress_midnight_3eca57dc`
| Detector | F1 | Leak-day recall | Leak-event recall | False alarms / leak-free day | Missed events |
|---|---|---|---|---|---|
| M1 raw | 0.829 | 0.703 | 0.775 | 0.722 [0.05, 2.14] | 9 |
| M2 recalibration | 0.299 | 0.154 | 0.350 | 0.015 | 26 |
| M3 v1 | **0.946** | **0.846** | **0.975** | **0.008** | **1** |
| M3 full | 0.917 | 0.824 | 0.925 | 0.481 [0.00, 1.51] | 3 |

### Stress: every site already leaking when observation starts (cold start) — `…m3v3_stress_coldstart_763d7329`
| Detector | F1 | Leak-day recall | False alarms / leak-free day | Missed events |
|---|---|---|---|---|
| M1 raw | 0.691 | 0.796 | 0.041 | 8 |
| M2 recalibration | 0.139 | 0.155 | 0.041 | 32 |
| M3 v1 | 0.946 | 0.951 | 0.033 | 2 |
| **M3 full** | **0.957** | **0.961** | **0.000** | 2 |

### Stress: heavy dropouts, fast drift, meter swaps, delay changes — `…m3v3_stress_sensors_54ce44af`
| Detector | F1 | Leak-day recall | False alarms / leak-free day | Missed events |
|---|---|---|---|---|
| M1 raw | 0.399 | 0.314 | 0.110 | 14 |
| M2 recalibration | 0.060 | 0.059 | 0.035 | 19 |
| M3 v1 | 0.894 | 0.824 | 0.075 | 2 |
| **M3 full** | **0.902** | **0.843** | 0.064 | 2 |

### Replay of the exact Milestone 2 failure dataset (seed 45; seen before, so a replay, not confirmatory)
| Detector | F1 | Leak-day recall | Days with a false alarm |
|---|---|---|---|
| M1 raw (`thr_raw`) | 0.913 | 0.980 | 0.00 |
| M2 (`thr_corr_persist`) | 0.614 | 0.531 | 0.05 |
| **M3 full** | **0.983** | **1.000** | **0.00** |

### Pre-registered verdicts, v3 (paired site-level bootstrap)
| ID | Claim | Δ [95% CI] | Verdict |
|---|---|---|---|
| R1 | M3 F1 > M2 (realistic) | +0.583 [+0.465, +0.694] | supported |
| R2 | M3 leak-day recall > M2 | +0.605 [+0.513, +0.679] | supported |
| R3 | M3 F1 > M1 | +0.244 [+0.082, +0.472] | supported |
| R4 | M3 faster than M1 (event delay) | −59 min [−121, +1] | inconclusive |
| R5 | ≤ +0.05 false alarms/leak-free day vs M2 | +0.011 [−0.041, +0.060] | inconclusive |
| R6 | ≤ +0.05 false alarms/leak-free day vs M1 | +0.017 [−0.022, +0.062] | inconclusive |
| R7 | Raw net alone > M2 leak-day recall | +0.395 [+0.238, +0.540] | supported |
| R8 | v3 adds ≤ 0.03 false alarms/leak-free day vs v1 | −0.028 [−0.068, 0.000] | supported |
| R9 | v3 loses ≤ 3 pts leak-day recall vs v1 | 0.000 [0, 0] | supported |
| N1 | Midnight: M3 leak-day recall > M2 | +0.670 [+0.607, +0.734] | supported |
| N2 | Midnight: M3 leak-event recall > M2 | +0.575 [+0.417, +0.718] | supported |
| N3 | Midnight: raw net alone > M2 | +0.352 [+0.212, +0.488] | supported |
| N4 | Midnight: ≤ +0.05 false alarms/leak-free day vs M2 | +0.466 [−0.016, +1.496] | inconclusive |
| C1 | Cold start: M3 > M2 leak-day recall | +0.806 [+0.737, +0.871] | supported |
| C2 | Cold start: M3 loses ≤ 5 pts vs M1 | +0.165 [+0.071, +0.261] | supported |
| C3 | Cold start: raw net alone > M2 | +0.738 [+0.632, +0.837] | supported |
| C4 | Cold start: v2 options > v1 | +0.010 [0.000, +0.032] | inconclusive |
| C5 | Cold start: ≤ +0.05 false alarms vs M1 | −0.041 [−0.074, −0.009] | supported |
| S1 | Sensor stress: M3 F1 > M2 | +0.842 [+0.748, +0.920] | supported |
| S2 | Sensor stress: M3 F1 > M1 | +0.503 [+0.297, +0.794] | supported |
| S3 | Sensor stress: ≤ +0.10 false alarms vs M1 | −0.046 [−0.127, +0.045] | supported |
| S4 | Sensor stress: ≤ +0.10 false alarms vs M2 | +0.029 [−0.025, +0.097] | supported |
| P1 | Replay: M3 leak-day recall > M2 | +0.469 [+0.328, +0.610] | supported |
| P2 | Replay: M3 loses ≤ 5 pts vs M1 | +0.020 [0.000, +0.067] | supported |
| P3 | Replay: M3 F1 > M2 | +0.368 [+0.246, +0.500] | supported |
| P4 | Replay: ≤ +3 pts days-with-false-alarm vs M2 | −0.050 [−0.100, −0.010] | supported |

## 5. Does the raw-signal safety net catch what recalibration misses?

On the **same test days**: of the leak days that Milestone 2's recalibration **missed**, how many did the
raw net catch (from `reports/milestone-3-summary.json`, generated by `tools/m3_summary.py`)?

| Scenario (v3) | Leak days M2 missed | Raw net alone caught | Full M3 caught | …by the raw-net component | …ONLY by raw net / cal alarm |
|---|---|---|---|---|---|
| Realistic | 26 | 17 | 26 | 21 | 0 |
| Midnight | 77 | 36 | 61 | 49 | 1 |
| Cold start | 87 | 76 | 83 | 80 | 2 |
| Sensor stress | 48 | 10 | 40 | 18 | 1 |

**Yes, demonstrably.** On its own, the raw net recovers 7–76 of the leak days recalibration missed in
every scenario and suite. N3 and C3 were supported in all three suites, and R7 in v2 and v3 (in v1, R7
was posed at event level and was inconclusive). **Inside the full system it is
mostly a second line of defence.** The carried-forward core already catches those days, so the raw net
(together with the calibration alarm) is decisive on only a few: up to 12 in the v1 cold-start suite,
and 0–2 in v3.

## 6. Did the fix cost false alarms? (trade-offs)

- **Realistic:** +0.011 (vs M2) and +0.017 (vs M1) false alarms per leak-free day. That is roughly
  one extra false alarm per 60–90 leak-free days, and the CIs include 0. Non-inferiority at the
  0.05 margin is *inconclusive* (R5, R6), not shown.
- **Replay of the M2 failure:** M3 had *fewer* false alarms than M2 (0.000 vs 0.118 per leak-free day;
  P4 supported).
- **Cold start and sensor stress:** M3 had no more false alarms than M1, and fewer in the cold-start
  scenario (C5, S3 supported).
- **Midnight stress: one bad site.** Site 57 (large real meter bias, about +1.7 L/min) gave a weak,
  ambiguous pressure reading, the calibration was flagged leak-like, and the **calibration alarm
  flapped 54 times in one day** because its release level has no hysteresis. That one day drives the
  0.481 mean (CI [0.00, 1.51]); excluding it, the scenario has 9 false alarms over 133 leak-free days.
  M1 has the same kind of outlier (site 10, 85 onsets over 3 days). **Unresolved; see §8.**
- **The v2 options are a trade-off, not a strict win.** They help cold start (false alarms 0.033 → 0.000)
  and sensor stress, but in the midnight scenario v1 without them is better (F1 0.946 vs 0.917, and no
  flapping). No single configuration dominates.
- **Meter swap with no pressure sensor:** the safe fallback alarms rather than re-baselining, so a
  meter swap produces false alarms until an operator re-baselines (or `rebaseline_after_days` > 0).
  This is deliberate and covered by `test_meter_swap_without_pressure_falls_back_to_alarm`.

## 7. Reproducibility and integrity

- `check-repro` REPRODUCED for Milestone 1, the Milestone 2 failure run, the M3 v1 confirmatory run, and
  the v3 realistic and cold-start runs. All new options are opt-in, so every earlier recorded run still
  reproduces.
- In the replay, the Milestone 2 baselines reproduced the original Milestone 2 numbers exactly
  (F1 0.963 and 0.621 under the Milestone 2 objective).
- The site-clustered bootstrap reproduces earlier single-day CIs bit-for-bit (verified on the M2 confirmatory run).
- `python -m genesis verify-ledger` → ok, with every run, iteration, verdict and repro check chained.
- 81 tests pass, including a regression test for every known failure (M1 peak-rate bug, M2 absorption,
  M3 dev meter-swap lock-up, M3 v1 cold-start absorption, M3 v2 alarm-count-blind objective).

## 8. Remaining limitations

1. **Synthetic only.** The pressure physics in the simulator is clean. Real pressure is affected by
   pumps, valves and network effects. Pressure corroboration is the key enabler here, and the most
   likely to weaken on real data.
2. **Calibration-alarm flapping:** it needs hysteresis (latch, then release below a lower level). This
   is the first fix for the next milestone.
3. **A leak present since installation, smaller than the site's bias spread, is undetectable.** No
   self-calibrating method can see it, and neither can M1 (e.g. a 2 L/min leak on a site with negative
   bias, seed 61 in `tests/test_milestone3.py`). It needs an external commissioning check.
4. **Slow drift vs slowly growing leak:** EWMA tracking can absorb a leak that grows slower than the
   expected drift. This is not stress-tested yet.
5. **Event-level recall is a weak metric for multi-day leaks:** all methods eventually alarm on most
   events (R1, R7, R8 inconclusive in v1). Day-level recall and delay are more informative.
6. **Forking paths:** the design changed twice after confirmatory results (v1→v2→v3). Each change was
   then confirmed on fresh seeds, but the overall process has more researcher degrees of freedom than
   a single pre-registered test.
7. Single pipe segment; no network topology, pump schedules or sensor faults other than gaps and drift.
