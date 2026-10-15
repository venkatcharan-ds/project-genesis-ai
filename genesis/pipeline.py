"""End-to-end GENESIS experiment pipeline.

problem -> data -> candidates -> bounded improvement (validation) ->
single held-out test evaluation -> failure analysis -> report + ledger
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from genesis.analysis import analyze_failures, missed_leak_events, safeguard_attribution
from genesis.config import config_hash
from genesis.data.sequences import SiteConfig, generate_sites, split_sites
from genesis.data.synthetic_leak import GeneratorConfig, dataset_fingerprint, generate_dataset, split_episodes
from genesis.detectors import build
from genesis.evaluation import bootstrap_ci, evaluate
from genesis.hypotheses import judge, validate
from genesis.improvement import improve
from genesis.ledger import EvidenceLedger
from genesis.report import render_report


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _fit_and_test(name: str, params: dict[str, Any], train, test, *, n_boot: int, seed: int,
                  score_from: int = 0) -> dict[str, Any]:
    det = build(name, **params)
    t0 = time.perf_counter()
    det.fit(train)
    fit_seconds = time.perf_counter() - t0
    m = evaluate(det, test, score_from=score_from)
    m["runtime"]["fit_seconds"] = fit_seconds
    m["timestep_ci95"] = bootstrap_ci(m["per_episode"], n_boot=n_boot, seed=seed)
    return m


def run_experiment(cfg: dict[str, Any], out_root: str | Path, ledger_path: str | Path) -> dict[str, Any]:
    # Validate everything up front so a bad config leaves no partial run or ledger entries.
    cand_ids = [c.get("id", c["name"]) for c in cfg["candidates"]]
    if len(set(cand_ids)) != len(cand_ids):
        raise ValueError(f"candidate ids must be unique: {cand_ids}")
    hypotheses = cfg.get("hypotheses", [])
    validate(hypotheses, set(cand_ids))

    chash = config_hash(cfg)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{cfg['experiment']['name']}_{chash[:8]}"
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger = EvidenceLedger(ledger_path)
    log = lambda kind, payload: ledger.append(run_id, kind, payload)  # noqa: E731

    env = {"python": platform.python_version(), "numpy": np.__version__,
           "scikit_learn": sklearn.__version__, "platform": platform.platform(), "git_commit": _git_commit()}
    log("run_started", {"config_hash": chash, "config": cfg, "environment": env})

    # 1. Data (synthetic) ------------------------------------------------------
    seed = cfg["experiment"]["seed"]
    data_cfg = dict(cfg["data"])
    if data_cfg.pop("generator", "single_day") == "multi_day":
        # Milestone 3: consecutive days per site; split by whole site.
        episodes = generate_sites(SiteConfig(**data_cfg), seed)
        splits = split_sites(episodes, cfg["split"], seed)
    else:
        episodes = generate_dataset(GeneratorConfig(**data_cfg), seed)
        splits = split_episodes(episodes, cfg["split"], seed)
    data_info = {name: {"episodes": len(eps), "samples": int(sum(len(e) for e in eps)),
                        "leak_episodes": sum(e.meta["has_leak"] for e in eps),
                        "positive_samples": int(sum(e.label.sum() for e in eps)),
                        "sites": len({e.meta.get("site_id", e.episode_id) for e in eps}),
                        "preexisting_leak_days": sum(bool(e.meta.get("preexisting_leak")) for e in eps),
                        "missing_minutes": int(sum(np.isnan(e.flow_in).sum() + np.isnan(e.flow_out).sum()
                                                   for e in eps)),
                        "fingerprint_sha256": dataset_fingerprint(eps)} for name, eps in splits.items()}
    log("dataset_generated", {"synthetic": True, "seed": seed, "splits": data_info})

    # Pre-register hypotheses BEFORE any tuning or testing (ledger order is the proof).
    if hypotheses:
        log("hypotheses_registered", {"hypotheses": hypotheses})

    # 2-3. Candidates: improve on validation, then test once on held-out ------
    imp = cfg["improvement"]
    score_from = int(cfg["evaluation"].get("score_from_minute", 0))
    objective_cfg = {"max_fpr": imp["max_fpr"], "fpr_penalty": imp["fpr_penalty"],
                     "max_day_fa_rate": imp.get("max_day_fa_rate", 1.0),
                     "day_fa_penalty": imp.get("day_fa_penalty", 0.0),
                     "max_fa_per_clean_day": imp.get("max_fa_per_clean_day", float("inf")),
                     "fa_count_penalty": imp.get("fa_count_penalty", 0.0)}
    results: dict[str, Any] = {}
    for cand, cid in zip(cfg["candidates"], cand_ids):
        name = cand["name"]
        initial = {**build(name).params, **cand.get("params", {})}
        search = improve(
            name, initial, cand.get("search_space", {}), splits["train"], splits["val"],
            max_iterations=imp["max_iterations"], max_evaluations=imp["max_evaluations"],
            min_delta=imp["min_delta"], score_from=score_from, **objective_cfg,
            log=lambda entry, c=cid: log("improvement_iteration", {"detector": c, **entry}),
        )
        # Held-out test: evaluate both the initial and the tuned configuration so the
        # effect of the improvement loop is itself measured, not assumed.
        boot = {"n_boot": cfg["evaluation"]["bootstrap_samples"], "seed": seed, "score_from": score_from}
        baseline_test = _fit_and_test(name, initial, splits["train"], splits["test"], **boot)
        tuned_test = _fit_and_test(name, search["best_params"], splits["train"], splits["test"], **boot)
        failures = analyze_failures(tuned_test["per_episode"])
        failures["missed_leak_events"] = missed_leak_events(tuned_test["per_episode"])
        attribution = safeguard_attribution(tuned_test["per_episode"])
        results[cid] = {"detector": name, "description": cand.get("description", ""),
                        "initial_params": initial, "improvement": search,
                         "test_initial": baseline_test, "test_tuned": tuned_test, "failure_analysis": failures,
                        "attribution": attribution}
        log("test_evaluation", {"detector": cid, "params": search["best_params"],
                                "timestep": tuned_test["timestep"], "event": tuned_test["event"],
                                "runtime": tuned_test["runtime"], "timestep_ci95": tuned_test["timestep_ci95"],
                                "initial_params_timestep": baseline_test["timestep"]})

    # 4. Judge pre-registered hypotheses on the test set (paired bootstrap) -----
    verdicts = []
    for h in hypotheses:
        v = judge(h, results, cfg["evaluation"]["bootstrap_samples"], seed)
        verdicts.append(v)
        log("hypothesis_verdict", v)

    # 5. Rank candidates by held-out F1 (reported, not used for any tuning) ----
    ranking = sorted(results, key=lambda n: results[n]["test_tuned"]["timestep"]["f1"] or 0.0, reverse=True)
    summary = {"run_id": run_id, "synthetic": True, "config_hash": chash, "environment": env,
               "score_from_minute": score_from, "data": data_info, "ranking_by_test_f1": ranking,
               "results": results, "hypotheses": verdicts}

    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    (run_dir / "report.md").write_text(render_report(cfg, summary))
    log("run_completed", {"run_dir": str(run_dir), "ranking_by_test_f1": ranking})
    summary["run_dir"] = str(run_dir)
    return summary
