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

## Key design decisions

- **Episode-level splits and bootstrap.** Samples within a day are autocorrelated. Splitting or resampling by timestep would leak information and produce falsely narrow CIs.
- **Objective = F1 − penalty × max(0, FPR − budget).** This encodes the operational constraint that false alarms are expensive. *Known gap:* it budgets per-sample FPR, while operators experience per-day false alarms (see results).
- **Causality is tested, not assumed.** `tests/test_detectors.py::test_detector_is_causal` perturbs the future and checks past alarms are unchanged.
- **Hash-chained ledger.** Cheap tamper-evidence, with no blockchain or database needed. `verify-ledger` detects edits and deletions.
- **Ledger and runs are committed.** Small JSON artifacts are the project's evidence, and git history plus the chain make them auditable.

## Simulator (synthetic) assumptions

One pipe segment, 1 sample/min, 24 h episodes. Diurnal demand with AR(1) noise; inlet and outlet meters with Gaussian noise and per-episode calibration bias; 0–2 min transport lag; optional legitimate demand spikes; burst (step) or gradual (ramp) leaks starting after a 2 h warm-up; pressure falls with flow² and leak rate. **It has no network topology, sensor dropouts, or real demand data**, so conclusions are about algorithm behaviour on this model only.

## Roadmap

**Milestone 2: close the loop on failures** (all testable on the current simulator)
- Turn the three untested hypotheses into new candidates: per-episode baseline recalibration, lag-compensated residual, trend/slope feature for gradual leaks.
- Add a per-day false-alarm budget to the objective and report.
- Add a "hypothesis → experiment → verdict" record type to the ledger.

**Milestone 3: agentic research and ideation**
- Optional LLM step (`ANTHROPIC_API_KEY`) that proposes candidates and hypotheses from the problem statement and failure analysis. Its outputs are logged verbatim to the ledger and always run through the same evaluation gate. No key means the step is skipped, never faked.
- Plugin interface so new problem domains (not just leak detection) supply a data adapter, candidates and metrics.

**Milestone 4: reality contact**
- Real-data loader (e.g. public water-network benchmarks such as BattLeDIM / L-Town) with its own `synthetic: false` path and data provenance in the ledger.
- Dashboard (Streamlit or Next.js) over `experiments/` for comparing runs.
