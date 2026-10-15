"""Cross-candidate evidence for the Milestone 3 report, computed from recorded runs.

For each Milestone 3 run: of the leak days Milestone 2's recalibrating detector MISSED,
how many did each Milestone 3 component (and the raw net alone) catch on the same days?
Writes reports/milestone-3-summary.json. Run: python tools/m3_summary.py
"""

import glob
import json
import os

out = {}
for path in sorted(glob.glob("experiments/runs/*_water_leak_m3*/results.json")):
    s = json.load(open(path))
    res = s["results"]
    m2 = "m2_corr" if "m2_corr" in res else "thr_corr_persist"
    m2_rows = {r["episode_id"]: r for r in res[m2]["test_tuned"]["per_episode"]}
    missed = {i for i, r in m2_rows.items() if r["leak_type"] and not r["detected"]}
    entry = {"run_id": s["run_id"], "m2_candidate": m2, "m2_missed_leak_days": len(missed),
             "hypotheses": {h["id"]: {"verdict": h["verdict"], "diff": h["diff"], "ci95": h["ci95"]}
                            for h in s.get("hypotheses", [])}}
    full = res.get("m3_full", {}).get("test_tuned", {}).get("per_episode", [])
    rescued = [r for r in full if r["episode_id"] in missed]
    if rescued and "components" in rescued[0]:
        c = lambda r, k: r["components"][f"{k}_detect"]  # noqa: E731
        entry["m3_full_on_m2_missed_days"] = {
            "caught_by_any": sum(r["detected"] for r in rescued),
            "caught_by_core": sum(c(r, "core") for r in rescued),
            "caught_by_raw_net": sum(c(r, "raw") for r in rescued),
            "caught_by_cal_alarm": sum(c(r, "cal") for r in rescued),
            "caught_ONLY_by_raw_net_or_cal": sum((c(r, "raw") or c(r, "cal")) and not c(r, "core")
                                                 for r in rescued)}
    if "s_raw_net" in res:
        raw_rows = {r["episode_id"]: r for r in res["s_raw_net"]["test_tuned"]["per_episode"]}
        entry["raw_net_alone_on_m2_missed_days"] = sum(raw_rows[i]["detected"] for i in missed)
    out[s["experiment"]["name"] if "experiment" in s else s["run_id"]] = entry

os.makedirs("reports", exist_ok=True)
with open("reports/milestone-3-summary.json", "w") as f:
    json.dump(out, f, indent=2)
for name, e in out.items():
    m = e.get("m3_full_on_m2_missed_days", {})
    print(f"{e['run_id'][17:]:45s} M2 missed {e['m2_missed_leak_days']:3d} | M3 caught {m.get('caught_by_any','-')} "
          f"(core {m.get('caught_by_core','-')}, raw {m.get('caught_by_raw_net','-')}, cal {m.get('caught_by_cal_alarm','-')}, "
          f"only raw/cal {m.get('caught_ONLY_by_raw_net_or_cal','-')}) | raw net ALONE caught {e.get('raw_net_alone_on_m2_missed_days','-')}")
