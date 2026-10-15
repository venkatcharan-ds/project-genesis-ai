import json

import numpy as np
import pytest

from genesis.evaluation import bootstrap_ci, confusion, objective, timestep_metrics
from genesis.ledger import EvidenceLedger


def test_metrics_hand_computed():
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    preds = np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
    c = confusion(labels, preds)
    assert c == {"tp": 2, "fp": 1, "tn": 4, "fn": 1}
    m = timestep_metrics(c)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 5)
    assert m["f1"] == pytest.approx(2 / 3)


def test_metrics_undefined_when_no_predictions():
    m = timestep_metrics({"tp": 0, "fp": 0, "tn": 10, "fn": 3})
    assert m["precision"] is None and m["recall"] == 0 and m["f1"] == 0.0


def test_objective_penalizes_fpr_over_budget():
    m = {"timestep": {"f1": 0.8, "false_positive_rate": 0.03}}
    assert objective(m, max_fpr=0.01, fpr_penalty=10) == pytest.approx(0.6)
    assert objective(m, max_fpr=0.05, fpr_penalty=10) == pytest.approx(0.8)


def test_bootstrap_ci_contains_point_estimate():
    rows = [{"confusion": {"tp": tp, "fp": fp, "tn": 100, "fn": 10 - tp}} for tp, fp in
            [(8, 1), (9, 0), (7, 2), (10, 1), (6, 0), (9, 3)]]
    ci = bootstrap_ci(rows, n_boot=500, seed=0)
    lo, hi = ci["recall"]
    assert lo <= 49 / 60 <= hi


def test_ledger_chain_and_tamper_detection(tmp_path):
    led = EvidenceLedger(tmp_path / "ledger.jsonl")
    for i in range(3):
        led.append("run", "step", {"i": i})
    assert led.verify()[0]

    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    tampered = json.loads(lines[1]); tampered["payload"]["i"] = 99
    lines[1] = json.dumps(tampered)
    (tmp_path / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    ok, msg = led.verify()
    assert not ok and "entry 1" in msg


def test_ledger_detects_deleted_entry(tmp_path):
    led = EvidenceLedger(tmp_path / "ledger.jsonl")
    for i in range(3):
        led.append("run", "step", {"i": i})
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    (tmp_path / "ledger.jsonl").write_text("\n".join([lines[0], lines[2]]) + "\n")
    assert not led.verify()[0]
