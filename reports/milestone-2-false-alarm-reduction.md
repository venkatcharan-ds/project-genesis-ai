# GENESIS Milestone 2 — Reducing False Alarms in Leak Detection

> **SYNTHETIC EXPERIMENTS — NOT PHYSICAL-WORLD VALIDATION.** Every number below comes from
> `genesis/data/synthetic_leak.py`. These results describe algorithm behaviour on a simplified
> simulator; nothing here shows how these detectors perform on real pipes or meters.

## Question

Milestone 1 found that false alarms came mostly from **meter calibration bias** and **meter
transport delay**. Milestone 2 tests three fixes. All of them are causal: they use only past data.

| Fix | Module | How it works |
|---|---|---|
| Meter-delay correction | `genesis/preprocess.py` | Each day, pick the lag (0–5 min) that minimises residual variance in a 120-min calibration window; delay the inlet signal by that lag. |
| Daily baseline recalibration | `genesis/preprocess.py` | Subtract the mean aligned residual of the same window (removes that day's meter bias). |
| Alarm persistence | `genesis/detectors/base.py` | Alarm only after *N* consecutive raw alarms. |
| Per-day false-alarm budget | `genesis/evaluation.py` | Tuning objective = F1 − 10·max(0, FPR − 1%) − 2·max(0, days-with-false-alarm − 5%). |

Estimator check on the simulator (400 days): the lag estimate was correct on **95%** of days. The bias error
std was **0.14 L/min**, against a true bias spread of 0.69 L/min.

## Method

- All candidates are scored **only after minute 120** of each day, so detectors that need the
  calibration window are compared with the baselines on identical samples.
- Hypotheses were **pre-registered**: written to the evidence ledger before any tuning or testing.
- Verdicts use a **paired episode-level bootstrap** (1000 resamples) on the held-out test days.
  A hypothesis is *supported* only if the 95% CI of the difference lies entirely on the predicted side.
- Four experiments: exploratory (seed 42), confirmatory on **fresh data** (seed 44), and two
  stress tests of the recalibration assumption (seeds 43, 45). Each has 400 simulated days, split
  200 train / 100 val / 100 test.

## Results

### 1. Exploratory run (seed 42): 11-candidate ablation — [`report`](../experiments/runs/20261015T050709Z_water_leak_m2_50808a1e/report.md)

| Candidate | Test F1 | Leak-event recall | All days w/ false alarm |
|---|---|---|---|
| thr_raw (M1 method) | 0.863 | 0.82 | 0.01 |
| thr_lag (delay correction only) | 0.871 | 0.82 | 0.01 |
| thr_recal (recalibration only) | 0.926 | 0.94 | 0.00 |
| thr_corr (both) | 0.944 | 0.94 | 0.01 |
| thr_corr_persist (+ persistence) | 0.949 | 0.94 | 0.04 |

Pre-registered verdicts: **3 supported, 6 inconclusive, 1 contradicted.**

- F1 gains from corrections were **supported** for threshold (+0.082, CI [0.041, 0.134]) and CUSUM
  (+0.078, CI [0.042, 0.125]). Threshold leak-event recall rose 0.82 → 0.94 (H8 supported).
- All five "fewer false-alarm days" hypotheses were **inconclusive or contradicted**. Isolation Forest
  had *more* false-alarm days after corrections (H3: +0.05, CI [0.01, 0.10], contradicted).

**Why the false-alarm hypotheses failed (lesson learned):** the tuner had already pushed the raw
baselines down to about 1% of days with a false alarm, at the cost of recall. So corrections
could not show up as fewer false alarms. Instead, the tuner spent the noise reduction on *sensitivity*.
The hypotheses asked the wrong question. They stay in the record unchanged, because rewriting them
after seeing test results would be p-hacking.

*Exploratory, not pre-registered:* in the threshold ablation, recalibration accounts for most of the
gain (0.863 → 0.926). Delay correction adds about +0.01 alone and +0.02 on top of recalibration.
Persistence never moved from its initial value of 5 during tuning, and showed no false-alarm benefit.

### 2. Confirmatory run on fresh data (seed 44) — [`report`](../experiments/runs/20261015T050941Z_water_leak_m2_confirm_76fbb1e4/report.md)

Reformulated question: *at a matched false-alarm budget, do corrections buy detection performance
without paying in false alarms?* Hypotheses were registered before this unseen dataset was generated and tested.

| Detector | Δ F1 | Δ leak-event recall | Δ mean delay (min) | Δ days w/ false alarm |
|---|---|---|---|---|
| Threshold | **+0.074** [0.035, 0.141] ✅ | **+0.143** [0.048, 0.244] ✅ | **−38** [−88, −3] ✅ | 0.000 [−0.04, 0.04] ⚪ |
| CUSUM | **+0.065** [0.028, 0.128] ✅ | **+0.143** [0.059, 0.241] ✅ | −18 [−48, +11] ⚪ | **−0.070** [−0.13, −0.02] ✅ |
| Isolation Forest | **+0.059** [0.030, 0.104] ✅ | +0.061 [0.000, 0.133] ⚪ | **−26** [−43, −10] ✅ | 0.000 [−0.05, 0.04] ⚪ |

✅ supported ⚪ inconclusive. **8 of 12 supported, 4 inconclusive, 0 contradicted.**

The F1 improvement replicated on unseen data for all three detector families. The "no false-alarm cost"
claims (margin 3 points) are mostly inconclusive. The point estimates are 0 or negative, but with 100
test days the CIs are ±4–5 points, too wide to confirm a 3-point margin.

### 3. Stress test A: leaks starting inside the calibration window (seed 43) — [`report`](../experiments/runs/20261015T050841Z_water_leak_m2_stress_calib_1558e717/report.md)

S1 and S2 (corrections lose ≤5 points of recall) were **supported**. **This test turned out to be too
weak, and we say so:** only 4 of 50 test leak days started inside the window, all at least 7 minutes in.
Recalibration absorbed only part of each leak, so it was never seriously challenged.

### 4. Stress test B: leaks already running at minute 0 (seed 45) — [`report`](../experiments/runs/20261015T051226Z_water_leak_m2_stress_preexisting_9b927739/report.md)

Every leak is already running when the day starts (e.g. it started overnight). We **pre-registered
predicted failures**.

| Detector | Raw leak-event recall | Corrected + persistence | Δ [95% CI] | Verdict |
|---|---|---|---|---|
| Threshold | 1.000 | 0.469 | −0.531 [−0.660, −0.392] | failure **confirmed** |
| CUSUM | 0.980 | 0.531 | −0.449 [−0.588, −0.294] | failure **confirmed** |
| Isolation Forest | 1.000 | 0.755 | −0.245 [−0.367, −0.122] | failure **confirmed** (smaller: pressure feature still sees the leak) |

Daily recalibration **absorbs a leak that is present during calibration into the baseline**. Burst
leaks that pre-exist become invisible. Gradual ones stay partly visible because they keep growing
after calibration. (Raw median delay reads 120 min only because scoring starts at minute 120.)

## Conclusions

1. **Supported (synthetic):** daily recalibration plus delay correction substantially improve detection
   (F1 +6–7 points, faster detection) at a matched false-alarm budget. This replicated on fresh data.
2. **Not supported:** that these fixes *reduce false-alarm days* at the operating points the tuner
   chooses. The benefit appears as sensitivity, not fewer alarms. Persistence showed no measurable benefit.
3. **Critical limitation:** naive daily recalibration fails on leaks that are already running at the start of
   the day. It **must not be deployed alone**. On this evidence, raw detectors are safer for those leaks.
4. Ranking within the corrected family (threshold vs CUSUM vs Isolation Forest) is **not settled**; CIs overlap.

## Reproducibility evidence

- `python -m genesis check-repro experiments/runs/20261015T045443Z_water_leak_v1_3e70a18c`: after the
  Milestone 2 refactor, Milestone 1 **reproduced exactly** (same data fingerprints, tuned parameters and
  test metrics). This is recorded as a `reproducibility_check` entry in the ledger.
- The exploratory run matched an earlier scratch run byte-for-byte on all deterministic metrics.
- `python -m genesis verify-ledger` → `ok (201 entries)` at the time of writing.

## Next (Milestone 3)

1. **Fix the critical limitation:** a *baseline guard*. Carry the bias estimate forward from previous
   days and flag days whose calibration bias is out of distribution versus history (a large pre-existing
   leak looks exactly like that). Run a raw detector in parallel as a safety net. Test with stress test B
   as the pre-registered gate.
2. Larger test sets, or sequential multi-day episodes, so the false-alarm-day CIs get narrow enough to
   test ±3-point claims.
3. A matched-operating-point evaluation that sweeps the alarm threshold on test and compares recall at a
   fixed false-alarm rate, instead of relying on the tuner to land on the same budget.
