# PROJECT GENESIS

**GENESIS is an AI-powered invention and engineering agent.** It takes a real-world problem, generates candidate solutions, builds prototypes, runs controlled experiments, compares them with measurable evidence, and improves them through bounded iterations, keeping a tamper-evident record of everything it did.

> **Status: Milestone 3 complete (on branch `milestone-3-leak-safe-calibration`).** Milestone 1 built the end-to-end
> loop; Milestone 2 added signal corrections and pre-registered hypotheses and exposed a critical failure;
> Milestone 3 adds leak-safe calibration, a raw-signal safety net and multi-day simulation, and fixes that failure.
> All results in this repo come from **synthetic data**. Nothing here is validated on physical pipes or sensors yet.

---

## What works today

```
problem (YAML) ──► synthetic data ──► 3 candidate detectors ──► bounded improvement loop (val split)
                                                                          │
   evidence ledger ◄── report.md + results.json ◄── failure analysis ◄── one-shot held-out test
   (hash-chained)
```

| Capability | Implementation |
|---|---|
| Accept a problem | `configs/water_leak.yaml` (problem statement, success criteria, candidates) |
| Generate solutions | Detector registry: mass-balance threshold, CUSUM, Isolation Forest |
| Prototype | `genesis/detectors/*`, all **causal** (no future data), enforced by tests |
| Controlled experiments | Seeded simulator, episode-level train/val/test split |
| Measurable comparison | Precision, recall, FPR, F1, event recall, detection delay, runtime, bootstrap 95% CIs |
| Identify failures | `genesis/analysis.py`: misses by leak type and size, false alarms vs confounders |
| Bounded improvement | Coordinate search with `max_iterations`, `max_evaluations`, `min_delta` hard limits |
| Reproducible evidence | Hash-chained `experiments/ledger.jsonl`, config hash, data SHA-256, git commit |

## Milestone 3 results (synthetic): [`full report`](reports/milestone-3-leak-safe-calibration.md)

**Problem:** Milestone 2's daily recalibration absorbs any leak that is already running in the morning calibration
window. On realistic multi-day data (leaks last 6–72 h), 52–60% of leak days start that way, and Milestone 2
caught only 6–44% of leak days.

**Fix (`leak_safe` detector):**
- a trusted per-site bias carried across days
- rejection of abnormal calibration windows
- pressure corroboration to tell a meter swap from a leak
- a calibration alarm instead of silent absorption
- an independent raw-signal safety net
- NaN-aware handling of sensor gaps
- configurable safe fallbacks while calibration is uncertain

Every safeguard can be switched on or off; with all of them off, `leak_safe` equals the Milestone 2 detector exactly.

Final confirmatory suite (v3: fresh seeds, hypotheses committed before running, 32 test sites × 7 days each):

| Scenario | F1: M1 / M2 / **M3** | Leak-day recall: M1 / M2 / **M3** | False alarms per leak-free day: M1 / M2 / **M3** |
|---|---|---|---|
| Realistic | 0.701 / 0.362 / **0.945** | 0.721 / 0.395 / **1.000** | 0.028 / 0.033 / 0.044 |
| Leaks at midnight | 0.829 / 0.299 / **0.917** | 0.703 / 0.154 / **0.824** | 0.722 / 0.015 / 0.481* |
| Already leaking at first observation | 0.691 / 0.139 / **0.957** | 0.796 / 0.155 / **0.961** | 0.041 / 0.041 / **0.000** |
| Heavy sensor stress | 0.399 / 0.060 / **0.902** | 0.314 / 0.059 / **0.843** | 0.110 / 0.035 / 0.064 |
| Exact M2 failure replay | 0.913 / 0.614 / **0.983** | 0.980 / 0.531 / **1.000** | 0.000 / 0.118 / **0.000** |

21 of 26 hypotheses supported, 5 inconclusive, 0 contradicted. \*One site-day accounts for 55 of the 64 alarm onsets:
the calibration alarm flaps because it has no hysteresis (unresolved; M1 shows a similar single-site outlier).
On its own, the raw safety net recovers 7–76 of the leak days Milestone 2 missed in every scenario. Inside the full
system it is mostly a second line behind the carried-forward core. The false-alarm cost on realistic data
(+0.01–0.02 per leak-free day) is *not* shown to be within a 0.05 margin (inconclusive). Getting here took three
confirmatory rounds: v1 had a cold-start gap, and v2 had a tuning objective that ignored the number of alarms.
Both are documented in the report.

## Milestone 2 results (synthetic): [`full report`](reports/milestone-2-false-alarm-reduction.md)

Fixes tested: per-day **meter-delay correction**, **daily baseline recalibration**, **alarm persistence**, and a
**per-day false-alarm budget** in the tuning objective.

- **Replicated on fresh data:** at a matched false-alarm budget, corrections raised test F1 by +0.074 / +0.065 / +0.059
  (threshold / CUSUM / Isolation Forest). All three paired-bootstrap CIs exclude 0. Overall, 8 of 12 confirmatory
  hypotheses were supported, 4 inconclusive, 0 contradicted.
- **Not supported:** that the fixes reduce *days with false alarms*. The tuner converts the noise reduction into
  sensitivity instead. Persistence showed no measurable benefit.
- **Critical limitation found and measured:** if a leak is already running when the day starts, daily
  recalibration absorbs it into the baseline. Leak-event recall dropped from 1.00 to 0.47 (threshold) and
  0.53 (CUSUM), and from 1.00 to 0.76 (Isolation Forest). Do not use recalibration alone.

## Milestone 1 results (synthetic, held-out test set: 60 episodes / 86,400 samples)

Full report: [`reports/milestone-1-water-leak-report.md`](reports/milestone-1-water-leak-report.md)

| Detector | Precision | Recall | FPR | F1 [95% CI] | Median delay | Leak-free days with a false alarm |
|---|---|---|---|---|---|---|
| Isolation Forest | 0.985 | 0.855 | 0.0035 | 0.915 [0.851, 0.956] | 10 min | 14.8% |
| CUSUM | 0.967 | 0.833 | 0.0076 | 0.895 [0.825, 0.943] | 3 min | 37.0% |
| Mass-balance threshold | 0.951 | 0.839 | 0.0117 | 0.892 [0.824, 0.940] | 20 min | 14.8% |

**What the evidence says, and what it doesn't:**
- The three detectors' F1 confidence intervals **overlap heavily**. On this data, Isolation Forest is *not* demonstrated to be better; the ranking is unsettled.
- **Trade-off:** CUSUM detects fastest (median 3 min) but raises false alarms on 37% of leak-free days, mostly on days with demand spikes (64% vs 19%).
- **The improvement loop helped on held-out data for all three** (test F1: threshold 0.811→0.892, CUSUM 0.850→0.895, IF 0.910→0.915). It tuned on validation only.
- **The operational bottleneck is false alarms per day**, not per-sample FPR. A per-sample FPR of 0.35% still means a false alarm on roughly 1 in 7 leak-free days.
- All detectors missed the same 3 gradual leaks. Two of them peaked at 0.41 and 0.67 L/min, below the residual noise floor (~1.4 L/min per sample).

## Quickstart

```bash
git clone https://github.com/venkatcharan-ds/project-genesis-ai.git
cd project-genesis-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m pytest -q                                                 # 81 tests
python -m genesis run --config configs/water_leak.yaml              # Milestone 1, ~15 s
python -m genesis run --config configs/water_leak_m2_confirm.yaml   # Milestone 2 confirmatory, ~1 min
python -m genesis run --config configs/water_leak_m3v3_confirm.yaml # Milestone 3 final confirmatory, ~1.5 min
python tools/make_m3_configs.py                                     # regenerate Milestone 3 configs
python tools/m3_summary.py                                          # cross-candidate safety-net evidence
python -m genesis check-repro experiments/runs/<run_id>             # re-run a recorded run, compare results
python -m genesis verify-ledger                                     # check evidence integrity
```

Each run writes `experiments/runs/<run_id>/{config.json,results.json,report.md}` and appends to `experiments/ledger.jsonl`.

## Repository layout

```
genesis/
  data/synthetic_leak.py     # simulator: independent days (SYNTHETIC)
  data/sequences.py          # M3: multi-day sites, drift, meter swaps, dropouts, multi-day leaks (SYNTHETIC)
  preprocess.py              # M2: causal meter-delay correction + daily bias recalibration
  detectors/                 # candidate solutions + registry
    base.py                  #   Detector interface, causal smoothing
    mass_balance.py          #   A: smoothed inflow-outflow threshold
    cusum.py                 #   B: one-sided CUSUM
    isolation_forest.py      #   C: unsupervised ML on flow + pressure residuals
    leak_safe.py             #   M3: leak-safe calibration + raw safety net (all safeguards switchable)
  evaluation.py              # metrics, event metrics, runtime, bootstrap CIs, objective
  improvement.py             # bounded improvement loop (validation only)
  analysis.py                # failure analysis + untested hypotheses
  hypotheses.py              # M2: pre-registered hypotheses, paired-bootstrap verdicts
  repro.py                   # M2: re-run a recorded experiment and diff the results
  ledger.py                  # hash-chained evidence ledger
  pipeline.py                # end-to-end orchestration
  report.py                  # markdown report (always carries the synthetic banner)
  cli.py                     # `python -m genesis run | verify-ledger | check-repro`
tools/                       # M3 config generator (holds the pre-registered hypotheses), evidence summary
configs/                     # experiment definitions (M1, M2, M3 dev / v1 / v2 / v3 suites)
experiments/                 # run artifacts + ledger (committed as evidence)
reports/                     # curated milestone reports
docs/architecture.md         # design + roadmap
tests/                       # unit, causality, reproducibility, end-to-end
```

## Principles

1. **No fabricated results.** Every number in a report is computed by code in this repo from a recorded run.
2. **Synthetic is labelled synthetic.** The config refuses to run unless `synthetic: true`, and every report carries the banner.
3. **Test set is sacred.** Tuning sees train/val only. Test is evaluated once per configuration, after tuning.
   Hypotheses are pre-registered in the ledger; hypotheses revised after seeing results are tested on fresh data.
4. **Bounded autonomy.** Improvement loops have hard iteration and evaluation budgets.
5. **Secrets stay out of Git.** Use `.env` (git-ignored). See `.env.example`. Milestone 1 needs no credentials.

## Roadmap

See [`docs/architecture.md`](docs/architecture.md#roadmap). Next up: alarm hysteresis (to fix the calibration-alarm
flapping), slow-leak vs drift stress tests, then the agentic research layer and real-data contact.

---
*Built with Claude Code.*
