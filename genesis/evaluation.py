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


def evaluate(detector: Detector, episodes: list[Episode], score_from: int = 0) -> dict[str, Any]:
    """Run a fitted detector on episodes; return metrics and per-episode outcomes.

    Only samples with t >= score_from are scored. Detectors still see the whole
    day (they need the calibration window); this only makes comparisons fair
    when some candidates cannot decide during calibration.
    """
    t0 = time.perf_counter()
    preds = [detector.predict(ep) for ep in episodes]
    predict_seconds = time.perf_counter() - t0
    s0 = int(score_from)

    per_episode, delays = [], []
    for ep, p in zip(episodes, preds):
        lab, pr = ep.label[s0:], p[s0:]
        row: dict[str, Any] = {"episode_id": ep.episode_id, "confusion": confusion(lab, pr),
                               "leak_type": ep.meta["leak_type"],
                               "leak_peak_rate": round(ep.meta["leak_peak_rate"], 3),
                               "meter_bias": round(ep.meta["meter_bias"], 3),
                               "demand_spike": ep.meta["demand_spike"],
                               # Any alarm on a leak-free scored sample = a false alarm that day.
                               "any_false_alarm": bool((pr & ~lab).any())}
        if ep.meta["has_leak"]:
            start = ep.meta["leak_start"]
            first_scored = max(start, s0)
            hits = np.flatnonzero(p[first_scored:])
            delay = int(first_scored + hits[0] - start) if hits.size else None
            row.update(detected=bool(hits.size), pre_leak_false_alarm=bool(p[s0:start].any()),
                       delay_minutes=delay)
            if delay is not None:
                delays.append(delay)
        else:
            row["false_alarm"] = row["any_false_alarm"]
        per_episode.append(row)

    m = metrics_from_rows(per_episode)
    n = sum(sum(r["confusion"].values()) for r in per_episode)
    return {
        "confusion": m["confusion"],
        "timestep": m["timestep"],
        "event": {
            "leak_episodes": m["leak_episodes"],
            "event_recall": m["event_recall"],
            "median_detection_delay_min": float(np.median(delays)) if delays else None,
            "mean_detection_delay_min": float(np.mean(delays)) if delays else None,
            "clean_episodes": m["clean_episodes"],
            "episode_false_alarm_rate": m["episode_false_alarm_rate"],
            "day_false_alarm_rate": m["day_false_alarm_rate"],
        },
        "score_from_minute": s0,
        "runtime": {"predict_seconds": predict_seconds, "samples": n,
                    "predict_us_per_sample": 1e6 * predict_seconds / n if n else None},
        "per_episode": per_episode,
    }


def metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode outcomes. Used for point estimates and bootstrap resamples."""
    c = {k: int(sum(r["confusion"][k] for r in rows)) for k in ("tp", "fp", "tn", "fn")}
    leak = [r for r in rows if r["leak_type"] is not None]
    clean = [r for r in rows if r["leak_type"] is None]
    return {
        "confusion": c,
        "timestep": timestep_metrics(c),
        "leak_episodes": len(leak),
        "clean_episodes": len(clean),
        "event_recall": _safe_div(sum(r["detected"] for r in leak), len(leak)),
        # Share of leak-free days with >=1 alarm (Milestone 1 headline metric).
        "episode_false_alarm_rate": _safe_div(sum(r["false_alarm"] for r in clean), len(clean)),
        # Share of ALL days with >=1 alarm on a leak-free sample (includes pre-leak alarms).
        "day_false_alarm_rate": _safe_div(sum(r.get("any_false_alarm", False) for r in rows), len(rows)),
        # Mean minutes from leak start to first alarm, over detected leaks only.
        "mean_detection_delay": (float(np.mean(d)) if (d := [r["delay_minutes"] for r in leak
                                                              if r.get("delay_minutes") is not None]) else None),
    }


def metric_value(rows: list[dict[str, Any]], metric: str) -> float | None:
    m = metrics_from_rows(rows)
    return m["timestep"][metric] if metric in m["timestep"] else m[metric]


BOOT_METRICS = ("precision", "recall", "false_positive_rate", "f1",
                "event_recall", "episode_false_alarm_rate", "day_false_alarm_rate", "mean_detection_delay")


def bootstrap_ci(per_episode: list[dict[str, Any]], n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict[str, list[float | None]]:
    """Episode-level bootstrap CIs.

    Resamples whole episodes (not timesteps), because samples within an episode
    are strongly correlated; timestep resampling would give falsely narrow CIs.
    """
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in BOOT_METRICS}
    for _ in range(n_boot):
        sample = [per_episode[i] for i in rng.integers(0, len(per_episode), len(per_episode))]
        for k in BOOT_METRICS:
            v = metric_value(sample, k)
            if v is not None:
                draws[k].append(v)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: ([float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v else [None, None])
            for k, v in draws.items()}


def paired_bootstrap_diff(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str,
                          n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """CI for metric(B) - metric(A) on the SAME test episodes (paired resampling).

    Pairing removes day-to-day difficulty variance shared by both candidates, so
    it detects real differences that two overlapping marginal CIs would hide.
    """
    a = {r["episode_id"]: r for r in rows_a}
    b = {r["episode_id"]: r for r in rows_b}
    if a.keys() != b.keys():
        raise ValueError("paired comparison needs both candidates evaluated on the same episodes")
    ids = sorted(a)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = [ids[i] for i in rng.integers(0, len(ids), len(ids))]
        va, vb = metric_value([a[i] for i in pick], metric), metric_value([b[i] for i in pick], metric)
        if va is not None and vb is not None:
            diffs.append(vb - va)
    pa, pb = metric_value(list(a.values()), metric), metric_value(list(b.values()), metric)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"metric": metric, "a": pa, "b": pb,
            "diff": None if pa is None or pb is None else pb - pa,
            "ci95": [float(np.percentile(diffs, lo)), float(np.percentile(diffs, hi))] if diffs else [None, None]}


def objective(metrics: dict[str, Any], max_fpr: float, fpr_penalty: float,
              max_day_fa_rate: float = 1.0, day_fa_penalty: float = 0.0) -> float:
    """Score used by the improvement loop.

    F1, penalized when timestep FPR exceeds its budget and (Milestone 2) when
    the share of days with a false alarm exceeds its budget. Defaults reproduce
    the Milestone 1 objective.
    """
    f1 = metrics["timestep"]["f1"] or 0.0
    fpr = metrics["timestep"]["false_positive_rate"] or 0.0
    day_fa = (metrics.get("event") or {}).get("day_false_alarm_rate") or 0.0
    return (f1 - fpr_penalty * max(0.0, fpr - max_fpr)
            - day_fa_penalty * max(0.0, day_fa - max_day_fa_rate))
