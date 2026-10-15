"""Pre-registered hypotheses and evidence-based verdicts.

Hypotheses are declared in the experiment config, written to the ledger BEFORE
any candidate is tuned or tested, and judged afterwards with a paired
episode-level bootstrap on the held-out test set:

    supported     the 95% CI of (candidate - baseline) lies entirely on the
                  predicted side (beyond the margin, for non-inferiority claims)
    contradicted  the 95% CI lies entirely on the opposite side
    inconclusive  anything else: the data cannot tell, and we say so
"""

from __future__ import annotations

from typing import Any

from genesis.evaluation import BOOT_METRICS, paired_bootstrap_diff

DIRECTIONS = ("increase", "decrease", "no_decrease", "no_increase")


def validate(hypotheses: list[dict[str, Any]], candidate_ids: set[str]) -> None:
    for h in hypotheses:
        missing = {"id", "statement", "baseline", "candidate", "metric", "direction"} - set(h)
        if missing:
            raise ValueError(f"hypothesis {h.get('id')} missing {sorted(missing)}")
        if h["direction"] not in DIRECTIONS:
            raise ValueError(f"hypothesis {h['id']}: direction must be one of {DIRECTIONS}")
        if h["metric"] not in BOOT_METRICS:
            raise ValueError(f"hypothesis {h['id']}: metric must be one of {BOOT_METRICS}")
        for role in ("baseline", "candidate"):
            if h[role] not in candidate_ids:
                raise ValueError(f"hypothesis {h['id']}: unknown {role} '{h[role]}'")


def judge(h: dict[str, Any], results: dict[str, Any], n_boot: int, seed: int) -> dict[str, Any]:
    d = paired_bootstrap_diff(results[h["baseline"]]["test_tuned"]["per_episode"],
                              results[h["candidate"]]["test_tuned"]["per_episode"],
                              h["metric"], n_boot=n_boot, seed=seed)
    lo, hi = d["ci95"]
    m = float(h.get("margin", 0.0))
    if lo is None:
        verdict = "inconclusive"
    elif h["direction"] == "increase":
        verdict = "supported" if lo > 0 else "contradicted" if hi < 0 else "inconclusive"
    elif h["direction"] == "decrease":
        verdict = "supported" if hi < 0 else "contradicted" if lo > 0 else "inconclusive"
    elif h["direction"] == "no_decrease":     # non-inferiority: not worse by more than margin
        verdict = "supported" if lo > -m else "contradicted" if hi < -m else "inconclusive"
    else:                                     # no_increase
        verdict = "supported" if hi < m else "contradicted" if lo > m else "inconclusive"
    return {**h, **d, "verdict": verdict}
