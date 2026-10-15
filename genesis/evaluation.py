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


def _predict_all(detector: Detector, episodes: list[Episode]) -> list[np.ndarray]:
    """Predict every episode; consecutive days of one site go through predict_sequence
    in day order (stateful detectors carry state across days). Output keeps input order."""
    groups: dict[Any, list[int]] = {}
    for i, ep in enumerate(episodes):
        groups.setdefault(ep.meta.get("site_id", ("ep", ep.episode_id)), []).append(i)
    if not hasattr(detector, "predict_sequence"):     # duck-typed predictors
        return [detector.predict(ep) for ep in episodes]
    preds: list[np.ndarray | None] = [None] * len(episodes)
    for idx in groups.values():
        idx = sorted(idx, key=lambda i: episodes[i].meta.get("day_index", 0))
        for i, p in zip(idx, detector.predict_sequence([episodes[i] for i in idx])):
            preds[i] = p
    return preds  # type: ignore[return-value]


def _onsets(mask: np.ndarray) -> int:
    """Number of distinct alarm segments starting inside `mask`."""
    return int(np.sum(mask & ~np.concatenate([[False], mask[:-1]])))


def evaluate(detector: Detector, episodes: list[Episode], score_from: int = 0) -> dict[str, Any]:
    """Run a fitted detector on episodes; return metrics and per-episode outcomes.

    Only samples with t >= score_from are scored. Detectors still see the whole
    day (they need the calibration window); this only makes comparisons fair
    when some candidates cannot decide during calibration.
    """
    if hasattr(detector, "components_"):
        detector.components_ = {}
    t0 = time.perf_counter()
    preds = _predict_all(detector, episodes)
    predict_seconds = time.perf_counter() - t0
    comps = getattr(detector, "components_", {})
    s0 = int(score_from)

    per_episode, delays = [], []
    for ep, p in zip(episodes, preds):
        lab, pr = ep.label[s0:], p[s0:]
        meta = ep.meta
        row: dict[str, Any] = {"episode_id": ep.episode_id, "confusion": confusion(lab, pr),
                               "leak_type": meta["leak_type"],
                               "leak_peak_rate": round(meta["leak_peak_rate"], 3),
                               "meter_bias": round(meta["meter_bias"], 3),
                               "demand_spike": meta["demand_spike"],
                               # Any alarm on a leak-free scored sample = a false alarm that day.
                               "any_false_alarm": bool((pr & ~lab).any()),
                               # Distinct false alarms: alarm segments that begin on a leak-free sample.
                               "fa_onsets": _onsets(pr & ~lab)}
        if "site_id" in meta:
            row.update(cluster=meta["site_id"], day_index=meta["day_index"],
                       preexisting_leak=meta.get("preexisting_leak", False),
                       leak_in_calibration=meta.get("leak_in_calibration", False),
                       missing_minutes=meta.get("missing_minutes", 0), meter_step=meta.get("meter_step", False),
                       lag_change=meta.get("lag_change", False))
        if meta["has_leak"]:
            start = meta["leak_start"]
            hits = np.flatnonzero(p & ep.label & (np.arange(len(p)) >= s0))
            delay = int(hits[0] - start) if hits.size else None
            row.update(detected=bool(hits.size), pre_leak_false_alarm=bool(p[s0:start].any()),
                       delay_minutes=delay)
            if "leak_event_id" in meta:
                row.update(leak_event_id=meta["leak_event_id"], leak_event_start_abs=meta["leak_event_start_abs"],
                           first_hit_abs=(meta["day_index"] * len(p) + int(hits[0])) if hits.size else None)
            if delay is not None:
                delays.append(delay)
        else:
            row["false_alarm"] = row["any_false_alarm"]
        if ep.episode_id in comps:
            c = comps[ep.episode_id]
            scored = np.arange(len(p)) >= s0
            row["components"] = {
                "status": c["status"], "z": round(c["z"], 3),
                **{f"{k}_detect": bool((c[k] & ep.label & scored).any()) for k in ("core", "raw", "cal")},
                **{f"{k}_false_alarm": bool((c[k] & ~ep.label & scored).any()) for k in ("core", "raw", "cal")}}
        per_episode.append(row)

    m = metrics_from_rows(per_episode)
    n = sum(sum(r["confusion"].values()) for r in per_episode)
    ev = event_delays(per_episode)
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
            "false_alarms_per_clean_day": m["false_alarms_per_clean_day"],
            "leak_events": m["leak_events"],
            "leak_event_recall": m["leak_event_recall"],
            "median_event_delay_min": float(np.median(ev)) if ev else None,
            "mean_event_delay_min": float(np.mean(ev)) if ev else None,
        },
        "score_from_minute": s0,
        "runtime": {"predict_seconds": predict_seconds, "samples": n,
                    "predict_us_per_sample": 1e6 * predict_seconds / n if n else None},
        "per_episode": per_episode,
    }


def _cluster(r: dict[str, Any]) -> Any:
    """Resampling unit: the site for multi-day data, else the single day."""
    return r.get("cluster", r["episode_id"])


def _events(rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """Leak events (a multi-day leak is ONE event). Single-day data: one event per leak day."""
    ev: dict[Any, dict[str, Any]] = {}
    for r in rows:
        if r["leak_type"] is None:
            continue
        key = (_cluster(r), r.get("leak_event_id", r["episode_id"]))
        e = ev.setdefault(key, {"detected": False, "first_hit": None,
                                "start": r.get("leak_event_start_abs"), "delay": None})
        if r["detected"]:
            e["detected"] = True
            hit = r.get("first_hit_abs")
            if hit is None:                       # single-day data: delay is day-relative
                e["delay"] = r.get("delay_minutes")
            elif e["first_hit"] is None or hit < e["first_hit"]:
                e["first_hit"] = hit
                e["delay"] = hit - e["start"]
    return ev


def event_delays(rows: list[dict[str, Any]]) -> list[float]:
    return [e["delay"] for e in _events(rows).values() if e["delay"] is not None]


STAT_KEYS = ("tp", "fp", "tn", "fn", "leak_days", "detected_days", "clean_days", "clean_fa_days",
             "days", "any_fa_days", "delay_sum", "delay_n", "events", "events_detected",
             "event_delay_sum", "event_delay_n", "clean_fa_onsets")


def _stats(rows: list[dict[str, Any]]) -> np.ndarray:
    """Additive sufficient statistics for every metric (so clusters can be summed)."""
    leak = [r for r in rows if r["leak_type"] is not None]
    clean = [r for r in rows if r["leak_type"] is None]
    d = [r["delay_minutes"] for r in leak if r.get("delay_minutes") is not None]
    ev = _events(rows).values()
    ed = [e["delay"] for e in ev if e["delay"] is not None]
    return np.array([
        *(sum(r["confusion"][k] for r in rows) for k in ("tp", "fp", "tn", "fn")),
        len(leak), sum(r["detected"] for r in leak), len(clean), sum(r["false_alarm"] for r in clean),
        len(rows), sum(r.get("any_false_alarm", False) for r in rows), sum(d), len(d),
        len(ev), sum(e["detected"] for e in ev), sum(ed), len(ed),
        sum(r.get("fa_onsets", 0) for r in clean)], dtype=float)


def _metrics_from_stats(v: np.ndarray) -> dict[str, Any]:
    s = dict(zip(STAT_KEYS, v))
    c = {k: int(s[k]) for k in ("tp", "fp", "tn", "fn")}
    return {
        "confusion": c,
        "timestep": timestep_metrics(c),
        "leak_episodes": int(s["leak_days"]),
        "clean_episodes": int(s["clean_days"]),
        # Day-level: share of leak days with >=1 alarm on a leak sample (Milestone 1-2 name).
        "event_recall": _safe_div(s["detected_days"], s["leak_days"]),
        # Share of leak-free days with >=1 alarm (Milestone 1 headline metric).
        "episode_false_alarm_rate": _safe_div(s["clean_fa_days"], s["clean_days"]),
        # Share of ALL days with >=1 alarm on a leak-free sample (includes pre-leak alarms).
        "day_false_alarm_rate": _safe_div(s["any_fa_days"], s["days"]),
        # Mean minutes from (day-relative) leak start to first alarm, detected leak days only.
        "mean_detection_delay": _safe_div(s["delay_sum"], s["delay_n"]),
        # Milestone 3: per leak EVENT (multi-day leaks count once) and alarm counts.
        "leak_events": int(s["events"]),
        "leak_event_recall": _safe_div(s["events_detected"], s["events"]),
        "mean_event_delay": _safe_div(s["event_delay_sum"], s["event_delay_n"]),
        "false_alarms_per_clean_day": _safe_div(s["clean_fa_onsets"], s["clean_days"]),
    }


def metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode outcomes. Used for point estimates and bootstrap resamples."""
    return _metrics_from_stats(_stats(rows))


def _pick(m: dict[str, Any], metric: str) -> float | None:
    return m["timestep"][metric] if metric in m["timestep"] else m[metric]


def metric_value(rows: list[dict[str, Any]], metric: str) -> float | None:
    return _pick(metrics_from_rows(rows), metric)


BOOT_METRICS = ("precision", "recall", "false_positive_rate", "f1",
                "event_recall", "episode_false_alarm_rate", "day_false_alarm_rate", "mean_detection_delay",
                "leak_event_recall", "mean_event_delay", "false_alarms_per_clean_day")


def _cluster_matrix(rows: list[dict[str, Any]], order: list[Any] | None = None) -> tuple[list[Any], np.ndarray]:
    groups: dict[Any, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(_cluster(r), []).append(r)
    keys = order if order is not None else list(groups)
    return keys, np.stack([_stats(groups[k]) for k in keys])


def bootstrap_ci(per_episode: list[dict[str, Any]], n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> dict[str, list[float | None]]:
    """Cluster bootstrap CIs.

    Resamples whole clusters: single days for independent-day data, whole SITES
    for multi-day data (a site's days share meters and multi-day leaks, so they
    are not independent). Timestep resampling would give falsely narrow CIs.
    """
    _, mat = _cluster_matrix(per_episode)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in BOOT_METRICS}
    for _ in range(n_boot):
        m = _metrics_from_stats(mat[rng.integers(0, len(mat), len(mat))].sum(axis=0))
        for k in BOOT_METRICS:
            v = _pick(m, k)
            if v is not None:
                draws[k].append(v)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: ([float(np.percentile(v, lo)), float(np.percentile(v, hi))] if v else [None, None])
            for k, v in draws.items()}


def paired_bootstrap_diff(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str,
                          n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, Any]:
    """CI for metric(B) - metric(A) on the SAME test clusters (paired resampling).

    Pairing removes difficulty variance shared by both candidates, so it detects
    real differences that two overlapping marginal CIs would hide.
    """
    if {r["episode_id"] for r in rows_a} != {r["episode_id"] for r in rows_b}:
        raise ValueError("paired comparison needs both candidates evaluated on the same episodes")
    keys = sorted({_cluster(r) for r in rows_a})
    _, ma = _cluster_matrix(rows_a, keys)
    _, mb = _cluster_matrix(rows_b, keys)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        va = _pick(_metrics_from_stats(ma[idx].sum(axis=0)), metric)
        vb = _pick(_metrics_from_stats(mb[idx].sum(axis=0)), metric)
        if va is not None and vb is not None:
            diffs.append(vb - va)
    pa, pb = metric_value(rows_a, metric), metric_value(rows_b, metric)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {"metric": metric, "a": pa, "b": pb,
            "diff": None if pa is None or pb is None else pb - pa,
            "ci95": [float(np.percentile(diffs, lo)), float(np.percentile(diffs, hi))] if diffs else [None, None]}


def objective(metrics: dict[str, Any], max_fpr: float, fpr_penalty: float,
              max_day_fa_rate: float = 1.0, day_fa_penalty: float = 0.0,
              max_fa_per_clean_day: float = float("inf"), fa_count_penalty: float = 0.0) -> float:
    """Score used by the improvement loop.

    F1, penalized when timestep FPR exceeds its budget and (Milestone 2) when
    the share of days with a false alarm exceeds its budget. Defaults reproduce
    the Milestone 1 objective.
    """
    f1 = metrics["timestep"]["f1"] or 0.0
    fpr = metrics["timestep"]["false_positive_rate"] or 0.0
    day_fa = (metrics.get("event") or {}).get("day_false_alarm_rate") or 0.0
    # Milestone 3: penalize the NUMBER of distinct false alarms per leak-free day. Without
    # it, a configuration whose alarm flaps on/off many times a day costs nothing extra.
    fa_count = (metrics.get("event") or {}).get("false_alarms_per_clean_day") or 0.0
    over = max(0.0, fa_count - max_fa_per_clean_day) if fa_count_penalty else 0.0
    return (f1 - fpr_penalty * max(0.0, fpr - max_fpr)
            - day_fa_penalty * max(0.0, day_fa - max_day_fa_rate)
            - fa_count_penalty * over)
