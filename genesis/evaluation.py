"""Evaluator: timestep-level and event-level metrics plus runtime measurement."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from genesis.data.synthetic_leak import Episode
from genesis.detectors.base import Detector


def _safe_div(a: float, b: float) -> float | None:
    return a / b if b else None


def confusion(labels: np.ndarray, preds: np.ndarray) -> dict[str, int]:
    labels, preds = labels.astype(bool), preds.astype(bool)
    return {
        "tp": int(np.sum(labels & preds)),
        "fp": int(np.sum(~labels & preds)),
        "tn": int(np.sum(~labels & ~preds)),
        "fn": int(np.sum(labels & ~preds)),
    }


def timestep_metrics(c: dict[str, int]) -> dict[str, float | None]:
    precision = _safe_div(c["tp"], c["tp"] + c["fp"])
    recall = _safe_div(c["tp"], c["tp"] + c["fn"])
    fpr = _safe_div(c["fp"], c["fp"] + c["tn"])
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    return {"precision": precision, "recall": recall, "false_positive_rate": fpr, "f1": f1}


def evaluate(detector: Detector, episodes: list[Episode]) -> dict[str, Any]:
    """Run a fitted detector on episodes; return metrics and per-episode outcomes."""
    t0 = time.perf_counter()
    preds = [detector.predict(ep) for ep in episodes]
    predict_seconds = time.perf_counter() - t0

    labels_all = np.concatenate([ep.label for ep in episodes])
    preds_all = np.concatenate(preds)
    c = confusion(labels_all, preds_all)

    per_episode, delays = [], []
    leak_eps = detected = clean_eps = false_alarm_eps = 0
    for ep, p in zip(episodes, preds):
        row: dict[str, Any] = {"episode_id": ep.episode_id, "confusion": confusion(ep.label, p),
                               "leak_type": ep.meta["leak_type"],
                               "leak_peak_rate": round(ep.meta["leak_peak_rate"], 3),
                               "meter_bias": round(ep.meta["meter_bias"], 3),
                               "demand_spike": ep.meta["demand_spike"]}
        if ep.meta["has_leak"]:
            leak_eps += 1
            start = ep.meta["leak_start"]
            pre_leak_alarm = bool(p[:start].any())
            hits = np.flatnonzero(p[start:])
            row.update(detected=bool(hits.size), pre_leak_false_alarm=pre_leak_alarm,
                       delay_minutes=int(hits[0]) if hits.size else None)
            if hits.size:
                detected += 1
                delays.append(int(hits[0]))
        else:
            clean_eps += 1
            row["false_alarm"] = bool(p.any())
            false_alarm_eps += int(row["false_alarm"])
        per_episode.append(row)

    n = len(labels_all)
    return {
        "confusion": c,
        "timestep": timestep_metrics(c),
        "event": {
            "leak_episodes": leak_eps,
            "event_recall": _safe_div(detected, leak_eps),
            "median_detection_delay_min": float(np.median(delays)) if delays else None,
            "mean_detection_delay_min": float(np.mean(delays)) if delays else None,
            "clean_episodes": clean_eps,
            "episode_false_alarm_rate": _safe_div(false_alarm_eps, clean_eps),
        },
        "runtime": {"predict_seconds": predict_seconds, "samples": n,
                    "predict_us_per_sample": 1e6 * predict_seconds / n if n else None},
        "per_episode": per_episode,
    }


def bootstrap_ci(per_episode: list[dict[str, Any]], n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict[str, list[float | None]]:
    """Episode-level bootstrap CIs for timestep metrics.

    Resamples whole episodes (not timesteps), because samples within an episode
    are strongly correlated; timestep resampling would give falsely narrow CIs.
    """
    keys = ("tp", "fp", "tn", "fn")
    mat = np.array([[r["confusion"][k] for k in keys] for r in per_episode], dtype=float)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {"precision": [], "recall": [], "false_positive_rate": [], "f1": []}
    for _ in range(n_boot):
        tot = mat[rng.integers(0, len(mat), len(mat))].sum(axis=0)
        m = timestep_metrics(dict(zip(keys, tot)))
        for k in draws:
            if m[k] is not None:
                draws[k].append(m[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: ([float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v else [None, None])
            for k, v in draws.items()}


def objective(metrics: dict[str, Any], max_fpr: float, fpr_penalty: float) -> float:
    """Score used by the improvement loop: F1, penalized when timestep FPR exceeds budget."""
    f1 = metrics["timestep"]["f1"] or 0.0
    fpr = metrics["timestep"]["false_positive_rate"] or 0.0
    return f1 - fpr_penalty * max(0.0, fpr - max_fpr)
