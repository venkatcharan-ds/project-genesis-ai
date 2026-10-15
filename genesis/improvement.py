"""Bounded improvement loop: coordinate search over a discrete parameter grid.

Hard bounds: `max_iterations` accepted steps and `max_evaluations` total
candidate evaluations. Selection uses the VALIDATION split only; the test
split is never touched here.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from genesis.data.synthetic_leak import Episode
from genesis.detectors import build
from genesis.evaluation import evaluate, objective


def _neighbors(params: dict[str, Any], space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """All parameter sets differing from `params` by one grid step in one dimension."""
    out = []
    for key, grid in space.items():
        if params[key] not in grid:
            continue
        i = grid.index(params[key])
        for j in (i - 1, i + 1):
            if 0 <= j < len(grid):
                out.append({**params, key: grid[j]})
    return out


def improve(
    detector_name: str,
    initial_params: dict[str, Any],
    search_space: dict[str, list[Any]],
    train: list[Episode],
    val: list[Episode],
    *,
    max_iterations: int,
    max_evaluations: int,
    min_delta: float,
    max_fpr: float,
    fpr_penalty: float,
    max_day_fa_rate: float = 1.0,
    day_fa_penalty: float = 0.0,
    score_from: int = 0,
    max_fa_per_clean_day: float = float("inf"),
    fa_count_penalty: float = 0.0,
    log: Callable[[dict[str, Any]], None] = lambda _: None,
) -> dict[str, Any]:
    cache: dict[tuple, dict[str, Any]] = {}

    def score(params: dict[str, Any]) -> dict[str, Any]:
        key = tuple(sorted(params.items()))
        if key not in cache:
            det = build(detector_name, **params)
            t0 = time.perf_counter()
            det.fit(train)
            fit_s = time.perf_counter() - t0
            m = evaluate(det, val, score_from=score_from)
            obj = objective(m, max_fpr, fpr_penalty, max_day_fa_rate, day_fa_penalty,
                            max_fa_per_clean_day, fa_count_penalty)
            cache[key] = {"params": params, "objective": obj,
                          "val_timestep": m["timestep"], "val_event": m["event"], "fit_seconds": fit_s}
        return cache[key]

    current = score(dict(initial_params))
    history = [{"iteration": 0, "accepted": True, "reason": "initial parameters", **current}]
    log(history[-1])
    stop_reason = f"reached max_iterations={max_iterations}"

    for it in range(1, max_iterations + 1):
        candidates = [p for p in _neighbors(current["params"], search_space)
                      if tuple(sorted(p.items())) not in cache]
        budget = max_evaluations - len(cache)
        if budget <= 0:
            stop_reason = f"evaluation budget exhausted ({max_evaluations})"
            break
        if not candidates:
            stop_reason = "no untried neighbors (local optimum on grid)"
            break
        scored = [score(p) for p in candidates[:budget]]
        best = max(scored, key=lambda r: r["objective"])
        gain = best["objective"] - current["objective"]
        accepted = gain >= min_delta
        history.append({"iteration": it, "accepted": accepted, "gain": gain,
                        "candidates_evaluated": len(scored),
                        "reason": "improved validation objective" if accepted
                        else f"best gain {gain:.4f} < min_delta {min_delta}", **best})
        log(history[-1])
        if not accepted:
            stop_reason = "converged (no candidate improved by min_delta)"
            break
        current = best

    return {"best_params": current["params"], "best_val_objective": current["objective"],
            "evaluations": len(cache), "stop_reason": stop_reason, "history": history}
