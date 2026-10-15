"""Tests for Milestone 2: signal corrections, persistence, scoring window, hypotheses, repro."""

import numpy as np
import pytest

from genesis.data.synthetic_leak import Episode, GeneratorConfig, generate_dataset
from genesis.detectors import REGISTRY, build
from genesis.detectors.base import require_persistence
from genesis.evaluation import evaluate, metric_value, objective, paired_bootstrap_diff
from genesis.hypotheses import judge, validate
from genesis.preprocess import calibrate, corrected_residual, estimate_lag
from genesis.repro import compare

CORR = {"lag_comp": True, "recalibrate": True}


def _ep(flow_in, flow_out, label=None):
    n = len(flow_in)
    label = np.zeros(n, bool) if label is None else label
    return Episode(0, np.asarray(flow_in, float), np.asarray(flow_out, float), np.full(n, 3.0),
                   label.astype(float), label, {"has_leak": False, "leak_type": None, "leak_peak_rate": 0.0,
                                                "meter_bias": 0.0, "demand_spike": False})


# -- corrections -----------------------------------------------------------------
def test_lag_estimation_recovers_known_delay():
    rng = np.random.default_rng(0)
    demand = 100 + np.cumsum(rng.normal(0, 1.0, 600))
    for lag in (0, 1, 3):
        out = np.concatenate([np.full(lag, demand[0]), demand[: len(demand) - lag]])
        fin = demand + rng.normal(0, 0.3, 600)
        fout = out + rng.normal(0, 0.3, 600)
        assert estimate_lag(fin, fout, calib=200, max_lag=5) == lag


def test_recalibration_removes_constant_bias():
    rng = np.random.default_rng(1)
    d = 100 + rng.normal(0, 1, 1000)
    ep = _ep(d + 2.5, d)                      # pure bias, no leak
    r = corrected_residual(ep, lag_comp=False, recalibrate=True, calib_minutes=120, max_lag=5)
    assert abs(ep.residual[120:].mean() - 2.5) < 1e-9
    assert abs(r[120:].mean()) < 1e-9


def test_calibration_estimates_on_simulator():
    eps = generate_dataset(GeneratorConfig(n_episodes=100), 3)
    cal = [calibrate(e, 120, True, True, 5) for e in eps]
    assert np.mean([c.lag == e.meta["lag_minutes"] for c, e in zip(cal, eps)]) > 0.85
    err = [c.bias - e.meta["meter_bias"] for c, e in zip(cal, eps)]
    assert np.std(err) < 0.5 * np.std([e.meta["meter_bias"] for e in eps])


def test_leak_in_calibration_window_is_absorbed():
    """Documents the key assumption: a leak present during calibration becomes invisible."""
    d = np.full(1000, 100.0)
    leak = np.full(1000, 5.0)                 # leak from minute 0
    ep = _ep(d + leak, d, leak > 0)
    r = corrected_residual(ep, lag_comp=True, recalibrate=True, calib_minutes=120, max_lag=5)
    assert abs(r[120:].mean()) < 1e-9         # the 5 L/min leak has vanished


def test_defaults_leave_residual_untouched(splits):
    ep = splits["test"][0]
    np.testing.assert_array_equal(build("cusum").residual(ep), ep.residual)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_corrected_detectors_are_causal(name, splits):
    det = build(name, **CORR, persistence=3).fit(splits["train"])
    ep = splits["test"][0]
    for cut in (60, 300):                     # inside and after the calibration window
        fut = Episode(ep.episode_id, ep.flow_in.copy(), ep.flow_out.copy(), ep.pressure.copy(),
                      ep.leak_rate, ep.label, ep.meta)
        fut.flow_in[cut:] += 50.0
        fut.flow_out[cut:] -= 7.0
        np.testing.assert_array_equal(det.predict(ep)[:cut], det.predict(fut)[:cut])


# -- persistence ------------------------------------------------------------------
def test_persistence_requires_consecutive_alarms():
    a = np.array([1, 1, 0, 1, 1, 1, 1, 0], bool)
    np.testing.assert_array_equal(require_persistence(a, 1), a)
    np.testing.assert_array_equal(require_persistence(a, 3), [0, 0, 0, 0, 0, 1, 1, 0])


# -- evaluation -------------------------------------------------------------------
def test_score_window_excludes_early_samples(splits):
    det = build("mass_balance_threshold", k=0.0).fit(splits["train"])  # alarms almost everywhere
    full = evaluate(det, splits["test"])
    late = evaluate(det, splits["test"], score_from=120)
    n = len(splits["test"])
    assert full["runtime"]["samples"] - late["runtime"]["samples"] == 120 * n
    assert late["score_from_minute"] == 120


def test_detection_delay_measured_from_true_leak_start():
    n = 600
    label = np.zeros(n, bool); label[100:] = True
    ep = _ep(np.full(n, 100.0), np.full(n, 100.0), label)
    ep.meta.update(has_leak=True, leak_type="burst", leak_start=100)

    class Fixed:
        def predict(self, e):
            p = np.zeros(n, bool); p[150:] = True
            return p
    r = evaluate(Fixed(), [ep], score_from=120)["per_episode"][0]
    assert r["detected"] and r["delay_minutes"] == 50


def test_objective_day_fa_penalty():
    m = {"timestep": {"f1": 0.9, "false_positive_rate": 0.0}, "event": {"day_false_alarm_rate": 0.15}}
    assert objective(m, 0.01, 10.0) == pytest.approx(0.9)                       # M1 default ignores it
    assert objective(m, 0.01, 10.0, max_day_fa_rate=0.05, day_fa_penalty=2.0) == pytest.approx(0.7)


def _rows(fa_flags, ids=None):
    return [{"episode_id": i, "confusion": {"tp": 0, "fp": int(f), "tn": 10, "fn": 0},
             "leak_type": None, "false_alarm": bool(f), "any_false_alarm": bool(f)}
            for i, f in zip(ids or range(len(fa_flags)), fa_flags)]


def test_paired_bootstrap_detects_consistent_difference():
    a, b = _rows([1] * 30 + [0] * 30), _rows([0] * 60)
    d = paired_bootstrap_diff(a, b, "day_false_alarm_rate", n_boot=300)
    assert d["diff"] == pytest.approx(-0.5) and d["ci95"][1] < 0
    assert metric_value(a, "episode_false_alarm_rate") == pytest.approx(0.5)


def test_paired_bootstrap_requires_same_episodes():
    with pytest.raises(ValueError):
        paired_bootstrap_diff(_rows([0, 1]), _rows([0, 1], ids=[5, 6]), "f1")


# -- hypotheses -------------------------------------------------------------------
def _results(a_rows, b_rows):
    return {"A": {"test_tuned": {"per_episode": a_rows}}, "B": {"test_tuned": {"per_episode": b_rows}}}


@pytest.mark.parametrize("direction,expected", [("decrease", "supported"), ("increase", "contradicted"),
                                                ("no_increase", "supported"), ("no_decrease", "contradicted")])
def test_verdicts(direction, expected):
    h = {"id": "h", "statement": "s", "baseline": "A", "candidate": "B",
         "metric": "day_false_alarm_rate", "direction": direction, "margin": 0.05}
    res = _results(_rows([1] * 30 + [0] * 30), _rows([0] * 60))
    assert judge(h, res, n_boot=300, seed=0)["verdict"] == expected


def test_identical_candidates_are_inconclusive():
    h = {"id": "h", "statement": "s", "baseline": "A", "candidate": "B",
         "metric": "day_false_alarm_rate", "direction": "decrease"}
    rows = _rows([1, 0] * 20)
    assert judge(h, _results(rows, rows), n_boot=200, seed=0)["verdict"] == "inconclusive"


def test_hypothesis_validation():
    good = {"id": "h", "statement": "s", "baseline": "A", "candidate": "B", "metric": "f1", "direction": "increase"}
    validate([good], {"A", "B"})
    with pytest.raises(ValueError):
        validate([{**good, "candidate": "Z"}], {"A", "B"})
    with pytest.raises(ValueError):
        validate([{**good, "direction": "sideways"}], {"A", "B"})
    with pytest.raises(ValueError):
        validate([{**good, "metric": "vibes"}], {"A", "B"})


# -- reproducibility ----------------------------------------------------------------
def test_repro_compare_flags_differences():
    base = {"data": {"test": {"fingerprint_sha256": "x"}},
            "results": {"c": {"improvement": {"best_params": {"k": 2.0}},
                              "test_tuned": {"timestep": {"f1": 0.9}, "event": {"event_recall": 0.8}}}}}
    assert compare(base, base) == []
    changed = {**base, "results": {"c": {"improvement": {"best_params": {"k": 3.0}},
                                         "test_tuned": {"timestep": {"f1": 0.7}, "event": {"event_recall": 0.8}}}}}
    diffs = compare(base, changed)
    assert any("tuned param k" in d for d in diffs) and any("f1" in d for d in diffs)
