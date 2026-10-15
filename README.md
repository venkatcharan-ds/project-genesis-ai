# PROJECT GENESIS

**GENESIS is an AI-powered invention and engineering agent.** It takes a real-world problem, generates candidate solutions, builds prototypes, runs controlled experiments, compares them with measurable evidence, and improves them through bounded iterations, keeping a tamper-evident record of everything it did.

> **Status: Milestone 2 complete.** Milestone 1 built the end-to-end loop; Milestone 2 added causal signal
> corrections, pre-registered hypotheses with paired-bootstrap verdicts, stress tests, and reproducibility checks.
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

python -m pytest -q                                                 # 52 tests
python -m genesis run --config configs/water_leak.yaml              # Milestone 1, ~15 s
python -m genesis run --config configs/water_leak_m2_confirm.yaml   # Milestone 2 confirmatory, ~1 min
python -m genesis check-repro experiments/runs/<run_id>             # re-run a recorded run, compare results
python -m genesis verify-ledger                                     # check evidence integrity
```

Each run writes `experiments/runs/<run_id>/{config.json,results.json,report.md}` and appends to `experiments/ledger.jsonl`.

## Repository layout

```
genesis/
  data/synthetic_leak.py     # simulator: flows, pressure, leaks, confounders (SYNTHETIC)
  preprocess.py              # M2: causal meter-delay correction + daily bias recalibration
  detectors/                 # candidate solutions + registry
    base.py                  #   Detector interface, causal smoothing
    mass_balance.py          #   A: smoothed inflow-outflow threshold
    cusum.py                 #   B: one-sided CUSUM
    isolation_forest.py      #   C: unsupervised ML on flow + pressure residuals
  evaluation.py              # metrics, event metrics, runtime, bootstrap CIs, objective
  improvement.py             # bounded improvement loop (validation only)
  analysis.py                # failure analysis + untested hypotheses
  hypotheses.py              # M2: pre-registered hypotheses, paired-bootstrap verdicts
  repro.py                   # M2: re-run a recorded experiment and diff the results
  ledger.py                  # hash-chained evidence ledger
  pipeline.py                # end-to-end orchestration
  report.py                  # markdown report (always carries the synthetic banner)
  cli.py                     # `python -m genesis run | verify-ledger`
configs/                     # experiment definitions (M1, M2 exploratory/confirm/stress)
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

See [`docs/architecture.md`](docs/architecture.md#roadmap). Next up (Milestone 3): a *baseline guard* that fixes the
pre-existing-leak failure, gated on the stress test that exposed it.

---
*Built with Claude Code.*
