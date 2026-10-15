import json

import pytest

from genesis.cli import main
from genesis.config import load_config
from genesis.improvement import improve
from genesis.ledger import EvidenceLedger
from genesis.pipeline import run_experiment
from tests.conftest import tiny_experiment_config


def test_improvement_loop_respects_bounds(splits):
    out = improve("mass_balance_threshold", {"window": 15, "k": 4.0},
                  {"window": [5, 10, 15, 30, 60], "k": [2.0, 3.0, 4.0, 5.0, 6.0]},
                  splits["train"], splits["val"], max_iterations=2, max_evaluations=5,
                  min_delta=0.0, max_fpr=0.01, fpr_penalty=10.0)
    assert out["evaluations"] <= 5
    assert len(out["history"]) - 1 <= 2
    # Accepted steps never decrease the validation objective.
    accepted = [h["objective"] for h in out["history"] if h["accepted"]]
    assert accepted == sorted(accepted)


def test_end_to_end_run_and_reproducibility(tmp_path):
    cfg = tiny_experiment_config()
    a = run_experiment(cfg, tmp_path / "runs", tmp_path / "ledger.jsonl")
    b = run_experiment(cfg, tmp_path / "runs2", tmp_path / "ledger.jsonl")

    # Artifacts exist and are labelled synthetic.
    report = (tmp_path / "runs" / a["run_id"] / "report.md").read_text()
    assert "SYNTHETIC EXPERIMENT" in report and "NOT PHYSICAL-WORLD VALIDATION" in report
    saved = json.loads((tmp_path / "runs" / a["run_id"] / "results.json").read_text())
    assert saved["synthetic"] is True

    # Same config + seed => identical data and metrics (runtime excluded).
    assert a["data"] == b["data"]
    for name in a["results"]:
        assert a["results"][name]["test_tuned"]["timestep"] == b["results"][name]["test_tuned"]["timestep"]
        assert a["results"][name]["improvement"]["best_params"] == b["results"][name]["improvement"]["best_params"]

    # Ledger records both runs and its chain is intact.
    led = EvidenceLedger(tmp_path / "ledger.jsonl")
    assert led.verify()[0]
    kinds = [e["kind"] for e in led.entries()]
    assert kinds.count("run_started") == 2 and kinds.count("run_completed") == 2


def test_config_must_be_marked_synthetic(tmp_path):
    p = tmp_path / "c.yaml"
    cfg = tiny_experiment_config(); cfg["experiment"]["synthetic"] = False
    import yaml
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="synthetic"):
        load_config(p)


def test_shipped_config_loads():
    cfg = load_config("configs/water_leak.yaml")
    assert cfg["experiment"]["synthetic"] is True


def test_cli_verify_ledger(tmp_path, capsys):
    EvidenceLedger(tmp_path / "l.jsonl").append("r", "k", {})
    assert main(["verify-ledger", "--ledger", str(tmp_path / "l.jsonl")]) == 0
    assert "ok" in capsys.readouterr().out
