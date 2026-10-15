"""Failure analysis: turn per-episode outcomes into evidence-backed findings.

Findings are computed from the data only. Hypotheses are rule-based and
explicitly labelled as hypotheses (untested until a follow-up experiment).
"""

from __future__ import annotations

from typing import Any


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def analyze_failures(per_episode: list[dict[str, Any]]) -> dict[str, Any]:
    leak = [r for r in per_episode if r["leak_type"] is not None]
    clean = [r for r in per_episode if r["leak_type"] is None]

    by_type = {}
    for kind in ("burst", "gradual"):
        rows = [r for r in leak if r["leak_type"] == kind]
        by_type[kind] = {"n": len(rows), "detected": sum(r["detected"] for r in rows),
                         "recall": _rate(sum(r["detected"] for r in rows), len(rows))}

    missed = sorted((r for r in leak if not r["detected"]), key=lambda r: r["leak_peak_rate"])
    smallest_detected = min((r["leak_peak_rate"] for r in leak if r["detected"]), default=None)

    fa = [r for r in clean if r["false_alarm"]]
    fa_spike = sum(r["demand_spike"] for r in fa)
    clean_spike = sum(r["demand_spike"] for r in clean)
    fa_bias = [abs(r["meter_bias"]) for r in fa]
    clean_bias = [abs(r["meter_bias"]) for r in clean]

    findings = {
        "recall_by_leak_type": by_type,
        "missed_events": [{k: r[k] for k in ("episode_id", "leak_type", "leak_peak_rate")} for r in missed],
        "smallest_detected_leak_lpm": smallest_detected,
        "false_alarm_episodes": len(fa),
        "false_alarm_rate_with_demand_spike": _rate(fa_spike, clean_spike),
        "false_alarm_rate_without_demand_spike": _rate(len(fa) - fa_spike, len(clean) - clean_spike),
        "mean_abs_meter_bias_false_alarm": sum(fa_bias) / len(fa_bias) if fa_bias else None,
        "mean_abs_meter_bias_all_clean": sum(clean_bias) / len(clean_bias) if clean_bias else None,
        "pre_leak_false_alarms": sum(r["pre_leak_false_alarm"] for r in leak),
    }

    hypotheses = []
    g, b = by_type["gradual"]["recall"], by_type["burst"]["recall"]
    if g is not None and b is not None and g < b:
        hypotheses.append(f"Gradual leaks are missed more often than bursts (recall {g:.2f} vs {b:.2f}); "
                          "a longer integration window or trend feature may help.")
    with_s, without_s = findings["false_alarm_rate_with_demand_spike"], findings["false_alarm_rate_without_demand_spike"]
    if with_s is not None and without_s is not None and with_s > without_s:
        hypotheses.append(f"False alarms are more frequent on episodes with demand spikes ({with_s:.2f} vs "
                          f"{without_s:.2f}); lag-compensating the outlet meter may reduce them.")
    fb, cb = findings["mean_abs_meter_bias_false_alarm"], findings["mean_abs_meter_bias_all_clean"]
    if fb is not None and cb is not None and fb > 1.25 * cb:
        hypotheses.append(f"False-alarm episodes have larger meter bias (|bias| {fb:.2f} vs {cb:.2f} L/min); "
                          "per-episode baseline recalibration may help.")
    findings["hypotheses_untested"] = hypotheses
    return findings


def safeguard_attribution(per_episode: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Which component of a leak_safe detector caught each leak day, and which caused
    false alarms. Returns None for detectors without component diagnostics."""
    rows = [r for r in per_episode if "components" in r]
    if not rows:
        return None
    leak = [r for r in rows if r["leak_type"] is not None]
    clean = [r for r in rows if r["leak_type"] is None]
    c = lambda r, k: r["components"][k]  # noqa: E731
    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r["components"]["status"]] = status_counts.get(r["components"]["status"], 0) + 1
    pre = [r for r in leak if r.get("preexisting_leak")]
    return {
        "leak_days": len(leak),
        "detected_by": {k: sum(c(r, f"{k}_detect") for r in leak) for k in ("core", "raw", "cal")},
        # Safety-net evidence: leak days the calibrated core missed but another component caught.
        "caught_only_by_raw_net": sum(c(r, "raw_detect") and not c(r, "core_detect") for r in leak),
        "caught_only_by_cal_alarm": sum(c(r, "cal_detect") and not c(r, "core_detect")
                                        and not c(r, "raw_detect") for r in leak),
        "preexisting_leak_days": len(pre),
        "preexisting_detected_by": {k: sum(c(r, f"{k}_detect") for r in pre) for k in ("core", "raw", "cal")},
        "clean_days": len(clean),
        "false_alarm_days_by": {k: sum(c(r, f"{k}_false_alarm") for r in clean) for k in ("core", "raw", "cal")},
        "calibration_status_counts": status_counts,
    }


def missed_leak_events(per_episode: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Leak EVENTS (multi-day leaks counted once) with no alarm on any leak sample."""
    events: dict[Any, dict[str, Any]] = {}
    for r in per_episode:
        if r["leak_type"] is None:
            continue
        key = (r.get("cluster", r["episode_id"]), r.get("leak_event_id", r["episode_id"]))
        e = events.setdefault(key, {"event": key[1], "leak_type": r["leak_type"], "days": 0, "detected": False,
                                    "peak_rate": 0.0, "preexisting_on_some_day": False})
        e["days"] += 1
        e["detected"] |= bool(r["detected"])
        e["peak_rate"] = max(e["peak_rate"], r["leak_peak_rate"])
        e["preexisting_on_some_day"] |= bool(r.get("preexisting_leak"))
    return sorted((e for e in events.values() if not e["detected"]), key=lambda e: e["peak_rate"])
