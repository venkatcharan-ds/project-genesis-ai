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
