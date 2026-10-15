"""Generate Milestone 3 experiment configs (keeps candidate definitions identical across runs).

Run: python tools/make_m3_configs.py   (writes configs/water_leak_m3_*.yaml)
"""

import yaml

OFF = {"carry_forward": False, "reject_abnormal": False, "cal_alarm": False,
       "raw_net": False, "robust_cal": False, "missing_hold": False, "pressure_check": False}
CORE = {"window": [15, 30, 60, 120], "k": [2.0, 3.0, 4.0, 5.0, 6.0], "persistence": [1, 5, 15]}
RAW = {"raw_k": [3.0, 4.0, 5.0, 6.0, 8.0]}


def cand(cid, desc, name, params, space):
    return {"id": cid, "name": name, "description": desc, "params": params, "search_space": space}


def single(flag, desc, extra_space=None):
    return cand(f"s_{flag}", desc, "leak_safe",
                {**OFF, flag: True, "window": 60, "k": 3.0, "persistence": 5},
                {**CORE, **(extra_space or {})})


CANDIDATES = [
    cand("m1_raw", "Milestone 1: raw mass-balance threshold", "mass_balance_threshold",
         {"window": 60, "k": 2.0}, {k: v for k, v in CORE.items() if k != "persistence"}),
    cand("m2_corr", "Milestone 2: daily recalibration + delay correction + persistence",
         "mass_balance_threshold", {"window": 60, "k": 3.0, "lag_comp": True, "recalibrate": True,
                                    "persistence": 5}, CORE),
    single("carry_forward", "Safeguard alone: trusted bias carried across days"),
    single("reject_abnormal", "Safeguard alone: reject abnormal calibration (fallback to prior)",
           {"tau_prior": [2.0, 3.0, 4.0]}),
    single("cal_alarm", "Safeguard alone: alarm on leak-like calibration", {"tau_prior": [2.0, 3.0, 4.0]}),
    single("raw_net", "Safeguard alone: independent raw-signal safety net", RAW),
    single("robust_cal", "Safeguard alone: median calibration + window quality check"),
    single("missing_hold", "Safeguard alone: NaN-aware evidence, hold alarm through gaps"),
    single("pressure_check", "Safeguard alone: pressure corroboration of calibration shifts",
           {"tau_pressure": [3.0, 4.0, 6.0]}),
    cand("m3_full", "Milestone 3: all safeguards", "leak_safe",
         {"window": 60, "k": 3.0, "persistence": 5},
         {**CORE, **RAW, "tau_shift": [3.0, 4.0, 6.0], "tau_prior": [2.0, 3.0, 4.0],
          "tau_pressure": [3.0, 4.0, 6.0]}),
]

REALISTIC = {"generator": "multi_day", "n_sites": 80, "days_per_site": 7,
             "drift_std_per_day": 0.05, "meter_step_prob_per_day": 0.02, "lag_change_prob_per_day": 0.03,
             "dropout_prob_per_day": 0.05, "demand_spike_probability": 0.3,
             "leak_rate_per_day": 0.12, "leak_start_mode": "uniform"}
IMPROVE = {"max_iterations": 8, "max_evaluations": 30, "min_delta": 0.002, "max_fpr": 0.01,
           "fpr_penalty": 10.0, "max_day_fa_rate": 0.05, "day_fa_penalty": 2.0}
SPLIT = {"train": 0.4, "val": 0.2, "test": 0.4}
EVAL = {"bootstrap_samples": 1000, "score_from_minute": 0}


def write(path, name, seed, statement, data, hypotheses=(), candidates=CANDIDATES, header=""):
    cfg = {"experiment": {"name": name, "synthetic": True, "seed": seed},
           "problem": {"statement": statement,
                       "success_criteria": {"max_timestep_fpr": 0.01, "max_day_false_alarm_rate": 0.05}},
           "data": data, "split": SPLIT, "candidates": list(candidates), "hypotheses": list(hypotheses),
           "evaluation": EVAL, "improvement": IMPROVE}
    with open(path, "w") as f:
        f.write(header + yaml.safe_dump(cfg, sort_keys=False, width=100))


if __name__ == "__main__":
    write("configs/water_leak_m3_dev.yaml", "water_leak_m3_dev", 301,
          "DEVELOPMENT run for Milestone 3 (multi-day sites, realistic mix of drift, meter swaps,\n"
          "delay changes, dropouts, spikes and multi-day leaks). Used to build and debug the\n"
          "safeguards. Confirmatory conclusions come only from fresh-seed runs.\n",
          REALISTIC,
          header="# GENESIS Milestone 3 DEVELOPMENT run (SYNTHETIC). Not used for confirmatory claims.\n")


# ---------------------------------------------------------------------------
# CONFIRMATORY SUITE. Hypotheses below were written and committed BEFORE any of
# these fresh seeds (302-305) were generated or run. Seed 45 replays the exact
# Milestone 2 failure dataset.
# ---------------------------------------------------------------------------
def H(i, st, a, b, m, d, margin=None):
    h = {"id": i, "statement": st, "baseline": a, "candidate": b, "metric": m, "direction": d}
    if margin is not None:
        h["margin"] = margin
    return h


CONFIRM_HYPS = [
    H("R1", "Full M3 detects more leak events than M2", "m2_corr", "m3_full", "leak_event_recall", "increase"),
    H("R2", "Full M3 has higher timestep F1 than M2", "m2_corr", "m3_full", "f1", "increase"),
    H("R3", "Full M3 has higher timestep F1 than M1", "m1_raw", "m3_full", "f1", "increase"),
    H("R4", "Full M3 detects leak events faster than M1", "m1_raw", "m3_full", "mean_event_delay", "decrease"),
    H("R5", "TRADE-OFF: M3 adds at most 0.05 false alarms per leak-free day vs M2",
      "m2_corr", "m3_full", "false_alarms_per_clean_day", "no_increase", 0.05),
    H("R6", "TRADE-OFF: M3 adds at most 0.05 false alarms per leak-free day vs M1",
      "m1_raw", "m3_full", "false_alarms_per_clean_day", "no_increase", 0.05),
    H("R7", "Raw safety net alone catches leak events that M2 recalibration misses",
      "m2_corr", "s_raw_net", "leak_event_recall", "increase"),
    H("R8", "Carry-forward alone detects more leak events than M2", "m2_corr", "s_carry_forward",
      "leak_event_recall", "increase"),
    H("R9", "Full M3 beats the raw safety net alone on F1", "s_raw_net", "m3_full", "f1", "increase"),
]

MIDNIGHT_HYPS = [
    H("N1", "Leaks active at midnight / starting during calibration: M3 detects more leak days than M2",
      "m2_corr", "m3_full", "event_recall", "increase"),
    H("N2", "Same scenario: M3 detects more leak events than M2", "m2_corr", "m3_full", "leak_event_recall", "increase"),
    H("N3", "Same scenario: raw safety net alone detects more leak days than M2", "m2_corr", "s_raw_net",
      "event_recall", "increase"),
    H("N4", "TRADE-OFF: M3 adds at most 0.05 false alarms per leak-free day vs M2",
      "m2_corr", "m3_full", "false_alarms_per_clean_day", "no_increase", 0.05),
]

COLD_HYPS = [
    H("C1", "Every site already leaking when observation starts (no clean history): M3 detects more "
      "leak days than M2", "m2_corr", "m3_full", "event_recall", "increase"),
    H("C2", "Same scenario: M3 loses at most 5 points of leak-day recall vs M1 raw", "m1_raw", "m3_full",
      "event_recall", "no_decrease", 0.05),
    H("C3", "Same scenario: raw safety net alone detects more leak days than M2", "m2_corr", "s_raw_net",
      "event_recall", "increase"),
]

SENSOR_HYPS = [
    H("S1", "Heavy sensor stress (dropouts, drift, meter swaps, delay changes): M3 F1 above M2",
      "m2_corr", "m3_full", "f1", "increase"),
    H("S2", "TRADE-OFF under sensor stress: M3 adds at most 0.10 false alarms per leak-free day vs M1",
      "m1_raw", "m3_full", "false_alarms_per_clean_day", "no_increase", 0.10),
    H("S3", "TRADE-OFF under sensor stress: M3 adds at most 0.10 false alarms per leak-free day vs M2",
      "m2_corr", "m3_full", "false_alarms_per_clean_day", "no_increase", 0.10),
    H("S4", "Under sensor stress, missing-data handling alone has fewer false alarms per leak-free "
      "day than M2", "m2_corr", "s_missing_hold", "false_alarms_per_clean_day", "decrease"),
]

REPLAY_HYPS = [
    H("P1", "Exact Milestone 2 failure dataset (leaks running from minute 0, single days): M3 detects "
      "more leak days than the M2 detector", "thr_corr_persist", "m3_full", "event_recall", "increase"),
    H("P2", "Same dataset: M3 loses at most 5 points of leak-day recall vs raw M1", "thr_raw", "m3_full",
      "event_recall", "no_decrease", 0.05),
    H("P3", "Same dataset: M3 F1 above the M2 detector", "thr_corr_persist", "m3_full", "f1", "increase"),
    H("P4", "TRADE-OFF: M3 adds at most 3 points of days-with-false-alarm vs M2", "thr_corr_persist",
      "m3_full", "day_false_alarm_rate", "no_increase", 0.03),
]


def write_confirmatory():
    hdr = ("# GENESIS Milestone 3 CONFIRMATORY run (SYNTHETIC). Fresh seed; hypotheses pre-registered\n"
           "# in tools/make_m3_configs.py and committed before this config was ever run.\n")
    write("configs/water_leak_m3_confirm.yaml", "water_leak_m3_confirm", 302,
          "CONFIRMATORY: realistic multi-day sites. Does leak-safe calibration fix the Milestone 2\n"
          "absorption failure without trading it for false alarms?\n", REALISTIC, CONFIRM_HYPS, header=hdr)
    write("configs/water_leak_m3_stress_midnight.yaml", "water_leak_m3_stress_midnight", 303,
          "STRESS (confirmatory): every leak starts between 23:00 and 02:00, so it is running at\n"
          "midnight or starts inside the morning calibration window.\n",
          {**REALISTIC, "leak_start_mode": "night", "leak_rate_per_day": 0.25}, MIDNIGHT_HYPS, header=hdr)
    write("configs/water_leak_m3_stress_coldstart.yaml", "water_leak_m3_stress_coldstart", 304,
          "STRESS (confirmatory): every site is already leaking when observation starts, so there\n"
          "is no clean history to carry forward (hardest case for carry-forward).\n",
          {**REALISTIC, "preexisting_at_start_prob": 1.0}, COLD_HYPS, header=hdr)
    write("configs/water_leak_m3_stress_sensors.yaml", "water_leak_m3_stress_sensors", 305,
          "STRESS (confirmatory): heavy sensor problems: dropouts, fast drift, frequent meter swaps\n"
          "and delay changes, more demand spikes. Measures the false-alarm cost of the safeguards.\n",
          {**REALISTIC, "dropout_prob_per_day": 0.3, "drift_std_per_day": 0.15,
           "meter_step_prob_per_day": 0.08, "lag_change_prob_per_day": 0.15,
           "demand_spike_probability": 0.5}, SENSOR_HYPS, header=hdr)

    # Exact replay of the Milestone 2 failure (same data, seed, split, objective, scoring window).
    m2 = yaml.safe_load(open("configs/water_leak_m2_stress_preexisting.yaml"))
    keep = [c for c in m2["candidates"] if c["id"] in ("thr_raw", "thr_corr_persist")]
    full = {**CANDIDATES[-1], "search_space": {k: v for k, v in CANDIDATES[-1]["search_space"].items()}}
    cfg = {**m2, "experiment": {"name": "water_leak_m3_replay_m2_failure", "synthetic": True, "seed": 45},
           "problem": {"statement": "REPLAY of the exact Milestone 2 failure dataset (seed 45, every leak running\n"
                                    "from minute 0, independent single days, scored from minute 120). Same\n"
                                    "baselines, objective and split as the Milestone 2 run, plus the M3 detector.\n",
                       "success_criteria": m2["problem"]["success_criteria"]},
           "candidates": keep + [full], "hypotheses": REPLAY_HYPS}
    with open("configs/water_leak_m3_replay_m2_failure.yaml", "w") as f:
        f.write(hdr + yaml.safe_dump(cfg, sort_keys=False, width=100))


if __name__ == "__main__":
    write_confirmatory()
