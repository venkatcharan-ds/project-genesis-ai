# GENESIS Architecture

## The loop

GENESIS turns "invent a solution" into a disciplined experimental loop:

| Stage | Module | Contract |
|---|---|---|
| 1. Problem intake | `configs/*.yaml`, `config.py` | Problem statement, success criteria, candidate list, budgets. Config is hashed. |
| 2. Data | `data/` | Seeded, fingerprinted (SHA-256). Must be labelled `synthetic: true` until a real-data loader exists. |
| 3. Candidate solutions | `detectors/` | Implement `Detector.fit(train)` / `predict(episode) -> bool[]`. Must be causal. Registered in `REGISTRY`. |
| 4. Improvement | `improvement.py` | Coordinate search on a discrete grid. Uses **val only**. Hard limits: `max_iterations`, `max_evaluations`, `min_delta`. |
| 5. Evaluation | `evaluation.py` | One held-out test pass per configuration (initial and tuned), so the loop's own effect is measured. |
| 6. Failure analysis | `analysis.py` | Computes where each solution fails. Emits hypotheses explicitly marked **untested**. |
| 7. Evidence | `ledger.py`, `report.py` | Append-only hash-chained JSONL plus a human report with a synthetic-data banner. |
| 8. Hypotheses | `hypotheses.py` | Declared in config, logged **before** tuning, judged on test by paired bootstrap: supported / contradicted / inconclusive. |
| 9. Reproducibility | `repro.py` | `check-repro` re-executes a recorded run in isolation and diffs every deterministic result. |

**Milestone 3: stateful, sequence-aware detection.** Data can be multi-day *sites* (`data/sequences.py`).
The evaluator sends each site's days, in order, through `Detector.predict_sequence`; stateless detectors
treat days independently. Splits are by site, and bootstrap CIs resample whole sites (a site's days share
meters and multi-day leaks). For single-day data this reduces bit-for-bit to the earlier per-day bootstrap.

`detectors/leak_safe.py` keeps per-site state across days:

```
morning calibration window ──► candidate (lag, bias) + pressure level
        │
        ▼
  compare to TRUSTED site state (or population prior on a cold start)
        │
        ├── OK ─────────────► update trusted bias/lag/pressure (EWMA)
        ├── METER_SHIFT ────► flow moved, site pressure did not → re-baseline
        ├── ABNORMAL_HIGH ──► leak-like → keep trusted baseline + calibration alarm
        ├── LOW_QUALITY / LEARNING_BLOCKED → keep trusted baseline
        ▼
  core: aligned residual − trusted bias ─► threshold ─┐
  raw safety net: raw residual (no calibration) ──────┼─► OR ─► hold through sensor gaps ─► alarm
  calibration alarm ──────────────────────────────────┘
```

The raw net uses a more sensitive threshold while calibration is uncertain (the first days of a site, or a
rejected calibration). Per-component alarms and the daily calibration status are recorded for attribution.

Signal corrections (`preprocess.py`) are shared options on every detector (`lag_comp`, `recalibrate`,
`persistence`). Their defaults reproduce Milestone 1 exactly, and this is verified by `check-repro`.

## Key design decisions

- **Episode-level splits and bootstrap.** Samples within a day are autocorrelated. Splitting or resampling by timestep would leak information and produce falsely narrow CIs.
- **Objective = F1 − penalty × max(0, FPR − budget).** This encodes the operational constraint that false alarms are expensive. Milestone 2 added an optional per-day false-alarm budget. Milestone 3 found that the objective still ignored the *number* of alarms (a flapping alarm cost nothing), and added a penalty on false alarms per leak-free day.
- **New behaviour is opt-in.** Every new option defaults to the previous behaviour, so `check-repro` keeps reproducing every recorded run.
- **Absorbing a leak is the costly error.** Ambiguous evidence (e.g. a weak pressure drop) is treated as leak-like, which trades a few false alarms for not silently erasing leaks.
- **Causality is tested, not assumed.** `tests/test_detectors.py::test_detector_is_causal` perturbs the future and checks past alarms are unchanged.
- **Hash-chained ledger.** Cheap tamper-evidence, with no blockchain or database needed. `verify-ledger` detects edits and deletions.
- **Ledger and runs are committed.** Small JSON artifacts are the project's evidence, and git history plus the chain make them auditable.

## Simulator (synthetic) assumptions

Milestone 3 adds `data/sequences.py`: continuous multi-day sites with persistent, drifting meter bias, meter swaps, mid-day delay changes, sensor dropouts, a pressure-sensor offset and drift, and leaks lasting 6–72 h that can start at night or before observation begins. Milestones 1–2 use `data/synthetic_leak.py`: one pipe segment, 1 sample/min, 24 h episodes. Diurnal demand with AR(1) noise; inlet and outlet meters with Gaussian noise and per-episode calibration bias; 0–2 min transport lag; optional legitimate demand spikes; burst (step) or gradual (ramp) leaks starting after a 2 h warm-up (or, for stress tests, already running at minute 0 via `preexisting_leak_share`); pressure falls with flow² and leak rate. **It has no network topology, sensor dropouts, or real demand data**, so conclusions are about algorithm behaviour on this model only.

## Roadmap

**Milestone 2: close the loop on failures** ✅ done. See `reports/milestone-2-false-alarm-reduction.md`.
- Delay correction, daily recalibration, persistence, per-day false-alarm budget, pre-registered hypotheses.
- Found and measured a critical failure: recalibration absorbs leaks that are already running when the day starts.

**Milestone 3: leak-safe calibration** ✅ done (branch `milestone-3-leak-safe-calibration`). See `reports/milestone-3-leak-safe-calibration.md`.
- The pre-existing-leak failure is fixed across four scenarios plus the exact M2 replay; the raw safety net is demonstrated; the false-alarm trade-offs are disclosed.

**Next (Milestone 3.x)**
- Alarm hysteresis for the calibration alarm (one site-day produced 54 alarm onsets).
- Stress test: a slowly growing leak against EWMA drift tracking.
- Threshold-sweep comparison at a fixed false-alarm rate; larger test sets to narrow the false-alarm CIs.

**Milestone 4: agentic research and ideation**
- Optional LLM step (`ANTHROPIC_API_KEY`) that proposes candidates and hypotheses from the problem statement and failure analysis. Its outputs are logged verbatim to the ledger and always run through the same evaluation gate. No key means the step is skipped, never faked.
- Plugin interface so new problem domains (not just leak detection) supply a data adapter, candidates and metrics.

**Milestone 5: reality contact**
- Real-data loader (e.g. public water-network benchmarks such as BattLeDIM / L-Town) with its own `synthetic: false` path and data provenance in the ledger.
- Dashboard (Streamlit or Next.js) over `experiments/` for comparing runs.
