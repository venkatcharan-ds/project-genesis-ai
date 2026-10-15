"""Reproducibility check: re-execute a recorded run and compare deterministic outputs.

Compared: dataset fingerprints, tuned parameters, and every held-out test metric
except wall-clock runtime. The re-run goes to a temporary directory and ledger,
and a single `reproducibility_check` entry is appended to the main ledger.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from genesis.ledger import EvidenceLedger
from genesis.pipeline import run_experiment


def _close(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return a is not None and b is not None and abs(a - b) <= tol
    return a == b


def compare(old: dict[str, Any], new: dict[str, Any], tol: float = 1e-12) -> list[str]:
    diffs = []
    for split, d in old["data"].items():
        if d["fingerprint_sha256"] != new["data"][split]["fingerprint_sha256"]:
            diffs.append(f"data fingerprint differs for split '{split}'")
    for cid, r in old["results"].items():
        if cid not in new["results"]:
            diffs.append(f"candidate '{cid}' missing from re-run")
            continue
        n = new["results"][cid]
        # Older runs may predate newly added (default-valued) params; compare the keys they recorded.
        for k, v in r["improvement"]["best_params"].items():
            if n["improvement"]["best_params"].get(k) != v:
                diffs.append(f"{cid}: tuned param {k} {v} -> {n['improvement']['best_params'].get(k)}")
        for section in ("timestep", "event"):
            for k, v in r["test_tuned"][section].items():
                if k in n["test_tuned"][section] and not _close(v, n["test_tuned"][section][k], tol):
                    diffs.append(f"{cid}: test {section}.{k} {v} -> {n['test_tuned'][section][k]}")
    return diffs


def check_run(run_dir: str | Path, ledger_path: str | Path) -> tuple[bool, list[str], dict[str, Any]]:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    old = json.loads((run_dir / "results.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        new = run_experiment(cfg, Path(tmp) / "runs", Path(tmp) / "ledger.jsonl")
    diffs = compare(old, new)
    entry = EvidenceLedger(ledger_path).append(new["run_id"], "reproducibility_check", {
        "original_run_id": old["run_id"], "original_git_commit": old["environment"]["git_commit"],
        "rerun_git_commit": new["environment"]["git_commit"], "reproduced": not diffs, "differences": diffs})
    return not diffs, diffs, entry
