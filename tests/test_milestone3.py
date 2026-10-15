"""Milestone 3: multi-day simulator, leak-safe calibration, safety net, regression tests."""

import numpy as np
import pytest

from genesis.data.sequences import DAY, SiteConfig, generate_sites, split_sites
from genesis.data.synthetic_leak import Episode, GeneratorConfig, dataset_fingerprint, generate_dataset, split_episodes
from genesis.detectors import build
from genesis.detectors.base import ffill, filled
from genesis.evaluation import evaluate, metrics_from_rows

OFF = dict(carry_forward=False, reject_abnormal=False, cal_alarm=False, raw_net=False,
           robust_cal=False, missing_hold=False, pressure_check=False)
M2 = dict(window=60, k=3.0, lag_comp=True, recalibrate=True, persistence=5)
M3 = dict(window=60, k=3.0, persistence=5)
QUIET = dict(leak_rate_per_day=1e-6, demand_spike_probability=0.0, meter_step_prob_per_day=0.0,
             lag_change_prob_per_day=0.0, dropout_prob_per_day=0.0)


@pytest.fixture(scope="module")
def train_sites():
    return generate_sites(SiteConfig(n_sites=12, days_per_site=4, leak_rate_per_day=0.1), seed=11)


def _clean_site(seed, days=4, **kw):
    return generate_sites(SiteConfig(n_sites=1, days_per_site=days, **{**QUIET, **kw}), seed=seed)


def _inject_leak(days, from_day, rate, start_minute=0):
    """Leak running from (from_day, start_minute) to the end, with its pressure signature."""
    out = []
    for d, ep in enumerate(days):
        leak = np.zeros(DAY)
        if d > from_day:
            leak[:] = rate
        elif d == from_day:
            leak[start_minute:] = rate
        demand = ep.flow_out
        dp = 0.4 * ((demand + leak) ** 2 - demand ** 2) / 100 ** 2 + 0.5 * leak / 100
        meta = {**ep.meta, "has_leak": bool(leak.any()), "leak_type": "burst" if leak.any() else None,
                "leak_start": int(np.flatnonzero(leak)[0]) if leak.any() else None,
                "leak_peak_rate": float(leak.max()), "leak_event_id": f"{ep.meta['site_id']}-x" if leak.any() else None,
                "leak_event_start_abs": from_day * DAY + start_minute,
                "preexisting_leak": bool(leak[0] > 0)}
        out.append(Episode(ep.episode_id, ep.flow_in + leak, ep.flow_out.copy(), ep.pressure - dp,
                           leak, leak > 0, meta))
    return out


# -- simulator ---------------------------------------------------------------------
def test_sites_deterministic_and_labelled():
    cfg = SiteConfig(n_sites=6, days_per_site=5, dropout_prob_per_day=0.2)
    a, b = generate_sites(cfg, 3), generate_sites(cfg, 3)
    assert dataset_fingerprint(a) == dataset_fingerprint(b)
    for ep in a:
        np.testing.assert_array_equal(ep.label, ep.leak_rate > 0)
        assert ep.meta["has_leak"] == bool(ep.label.any())
    assert any(np.isnan(e.flow_in).any() or np.isnan(e.flow_out).any() for e in a)


def test_multi_day_leaks_and_preexisting_at_start():
    eps = generate_sites(SiteConfig(n_sites=20, days_per_site=5, preexisting_at_start_prob=1.0,
                                    leak_duration_hours=(30, 72)), 4)
    day0 = [e for e in eps if e.meta["day_index"] == 0]
    assert all(e.label[0] and e.meta["preexisting_leak"] for e in day0)
    # A leak lasting >24 h appears on consecutive days under one event id.
    ids = [e.meta["leak_event_id"] for e in eps if e.meta["has_leak"]]
    assert len(set(ids)) < len(ids)


def test_night_mode_leaks_hit_calibration_window():
    eps = generate_sites(SiteConfig(n_sites=30, days_per_site=6, leak_start_mode="night",
                                    leak_rate_per_day=0.3), 5)
    starts = [e for e in eps if e.meta["has_leak"] and not e.meta["preexisting_leak"]]
    # 23:00-02:00: either inside the next morning's calibration window, or late on the day before.
    assert starts and all(e.meta["leak_start"] < 120 or e.meta["leak_start"] >= DAY - 60 for e in starts)


def test_split_by_site_is_disjoint():
    eps = generate_sites(SiteConfig(n_sites=10, days_per_site=3), 6)
    sp = split_sites(eps, {"train": 0.5, "val": 0.2, "test": 0.3}, 6)
    sites = {k: {e.meta["site_id"] for e in v} for k, v in sp.items()}
    assert not (sites["train"] & sites["test"]) and not (sites["val"] & sites["test"])
    assert sum(len(v) for v in sp.values()) == 30


# -- missing data ------------------------------------------------------------------
def test_ffill_is_causal_and_gap_free_episode_untouched():
    x = np.array([np.nan, 2.0, np.nan, 4.0])
    np.testing.assert_array_equal(ffill(x, lead=9.0), [9.0, 2.0, 2.0, 4.0])
    ep = generate_dataset(GeneratorConfig(n_episodes=1), 0)[0]
    assert filled(ep) is ep


def test_legacy_detectors_survive_sensor_gaps(train_sites):
    gappy = generate_sites(SiteConfig(n_sites=2, days_per_site=2, dropout_prob_per_day=1.0), 9)
    for name in ("mass_balance_threshold", "cusum", "isolation_forest"):
        det = build(name).fit(train_sites)
        for ep in gappy:
            p = det.predict(ep)
            assert p.dtype == bool and p.shape == ep.label.shape


def test_missing_hold_gives_no_new_evidence_during_gap(train_sites):
    det = build("leak_safe", **M3).fit(train_sites)
    days = _clean_site(21, days=2)
    gap_day = days[1]
    gap_day.flow_out[600:780] = np.nan
    preds = det.predict_sequence(days)
    assert not preds[1][600:780].any()


# -- equivalence & state -------------------------------------------------------------
def test_all_safeguards_off_equals_milestone2():
    eps = generate_dataset(GeneratorConfig(n_episodes=24), 3)
    sp = split_episodes(eps, {"train": 0.5, "val": 0.25, "test": 0.25}, 3)
    a = build("mass_balance_threshold", **M2).fit(sp["train"])
    b = build("leak_safe", **{**M3, **OFF}).fit(sp["train"])
    for ep in sp["test"]:
        np.testing.assert_array_equal(a.predict(ep), b.predict(ep))


def test_leak_safe_is_causal_across_days(train_sites):
    det = build("leak_safe", **M3).fit(train_sites)
    days = _clean_site(22, days=3)
    base = det.predict_sequence(days)
    future = [Episode(e.episode_id, e.flow_in.copy(), e.flow_out.copy(), e.pressure.copy(),
                      e.leak_rate, e.label, e.meta) for e in days]
    future[2].flow_in[300:] += 40.0
    future[2].pressure[300:] -= 0.5
    pert = det.predict_sequence(future)
    np.testing.assert_array_equal(base[0], pert[0])
    np.testing.assert_array_equal(base[1], pert[1])
    np.testing.assert_array_equal(base[2][:300], pert[2][:300])


def test_no_state_leaks_between_sites(train_sites):
    det = build("leak_safe", **M3).fit(train_sites)
    a = _inject_leak(_clean_site(23), from_day=1, rate=8.0)
    b = _clean_site(24)
    for ep in b:
        ep.meta["site_id"] = 99
    together = evaluate(det, a + b)["per_episode"][len(a):]
    alone = evaluate(det, b)["per_episode"]
    assert [r["confusion"] for r in together] == [r["confusion"] for r in alone]


# -- REGRESSION: the Milestone 2 failure ----------------------------------------------
def test_regression_m2_absorbs_leak_running_at_midnight(train_sites):
    """Milestone 2 failure, kept as a test so it is never hidden: a leak running when the
    day starts is absorbed by daily recalibration."""
    days = _inject_leak(_clean_site(31), from_day=2, rate=6.0)
    m2 = build("mass_balance_threshold", **M2).fit(train_sites)
    assert m2.predict(days[3])[120:].mean() < 0.2          # day 3: leak running since midnight


def test_regression_leak_safe_catches_leak_running_at_midnight(train_sites):
    days = _inject_leak(_clean_site(31), from_day=2, rate=6.0)
    det = build("leak_safe", **M3).fit(train_sites)
    preds = det.predict_sequence(days)
    assert preds[3][120:].mean() > 0.8                     # carried-forward baseline keeps it visible
    assert det.components_[days[3].episode_id]["status"] in ("ABNORMAL_HIGH", "LEARNING_BLOCKED")


def test_regression_leak_starting_during_calibration(train_sites):
    days = _inject_leak(_clean_site(32), from_day=1, rate=6.0, start_minute=45)
    det = build("leak_safe", **M3).fit(train_sites)
    preds = det.predict_sequence(days)
    assert preds[1][120:].mean() > 0.8 and preds[2].mean() > 0.8


def test_regression_cold_start_preexisting_leak(train_sites):
    """No history at all (first day observed) and the leak is already running."""
    days = _inject_leak(_clean_site(33, days=1), from_day=0, rate=8.0)
    m2 = build("mass_balance_threshold", **M2).fit(train_sites)
    det = build("leak_safe", **M3).fit(train_sites)
    assert m2.predict(days[0])[120:].mean() < 0.2
    assert det.predict_sequence(days)[0][120:].mean() > 0.8


def test_raw_net_alone_catches_what_recalibration_absorbs(train_sites):
    days = _inject_leak(_clean_site(34, days=1), from_day=0, rate=10.0)
    only_raw = build("leak_safe", **{**M3, **OFF, "raw_net": True, "raw_k": 5.0}).fit(train_sites)
    no_net = build("leak_safe", **{**M3, **OFF}).fit(train_sites)
    assert no_net.predict(days[0])[120:].mean() < 0.2
    p = only_raw.predict(days[0])
    assert p[120:].mean() > 0.8
    assert only_raw.components_[days[0].episode_id]["raw"][120:].mean() > 0.8


# -- REGRESSION: meter shift lock-up found during development --------------------------
def _meter_step(days, from_day, step):
    out = []
    for d, ep in enumerate(days):
        fin = ep.flow_in + (step if d >= from_day else 0.0)
        out.append(Episode(ep.episode_id, fin, ep.flow_out, ep.pressure, ep.leak_rate, ep.label, ep.meta))
    return out


def test_regression_meter_swap_is_rebaselined_when_pressure_normal(train_sites):
    days = _meter_step(_clean_site(41, days=5), from_day=2, step=2.5)
    det = build("leak_safe", **M3).fit(train_sites)
    preds = det.predict_sequence(days)
    assert det.components_[days[2].episode_id]["status"] == "METER_SHIFT"
    assert preds[3].sum() == 0 and preds[4].sum() == 0     # no multi-day false-alarm lock-up


def test_meter_swap_without_pressure_falls_back_to_alarm(train_sites):
    """Documented trade-off: with no pressure sensor a meter swap cannot be told from a
    leak, so the safe fallback alarms rather than silently re-baselining."""
    days = _meter_step(_clean_site(41, days=5), from_day=2, step=2.5)
    for ep in days:
        ep.pressure[:] = np.nan
    det = build("leak_safe", **M3).fit(train_sites)
    preds = det.predict_sequence(days)
    assert det.components_[days[2].episode_id]["status"] == "ABNORMAL_HIGH"
    assert preds[2][120:].mean() > 0.5


def test_rebaseline_after_days_option(train_sites):
    days = _meter_step(_clean_site(42, days=6), from_day=1, step=2.5)
    for ep in days:
        ep.pressure[:] = np.nan
    det = build("leak_safe", **M3, rebaseline_after_days=2).fit(train_sites)
    det.predict_sequence(days)
    statuses = [det.components_[e.episode_id]["status"] for e in days]
    assert "REBASELINED" in statuses


# -- evaluation ------------------------------------------------------------------------
def test_multi_day_leak_counts_as_one_event(train_sites):
    days = _inject_leak(_clean_site(51), from_day=1, rate=8.0)
    det = build("leak_safe", **M3).fit(train_sites)
    m = evaluate(det, days)
    assert m["event"]["leak_events"] == 1 and m["event"]["leak_episodes"] == 3
    assert m["event"]["leak_event_recall"] == 1.0
    assert m["event"]["mean_event_delay_min"] is not None


def test_false_alarm_onsets_counted_per_clean_day():
    rows = [{"episode_id": 1, "confusion": {"tp": 0, "fp": 5, "tn": 10, "fn": 0}, "leak_type": None,
             "false_alarm": True, "any_false_alarm": True, "fa_onsets": 3},
            {"episode_id": 2, "confusion": {"tp": 0, "fp": 0, "tn": 10, "fn": 0}, "leak_type": None,
             "false_alarm": False, "any_false_alarm": False, "fa_onsets": 0}]
    assert metrics_from_rows(rows)["false_alarms_per_clean_day"] == pytest.approx(1.5)


# -- REGRESSION: cold-start absorption found in the first confirmatory run (v1 -> v2) ----
V2 = dict(raw_k_uncertain=3.0, meter_shift_requires_site_pressure=True)


def test_regression_v1_cold_start_meter_shift_absorbs_leak(train_sites):
    """v1 failure kept as a test: with no site history, a leak running since day 0 was
    called a METER_SHIFT against the population pressure prior and absorbed on day 1."""
    days = _inject_leak(_clean_site(61, days=3), from_day=0, rate=3.5)
    v1 = build("leak_safe", **M3).fit(train_sites)
    p = v1.predict_sequence(days)
    assert p[0][120:].mean() > 0.8 and p[1][120:].mean() < 0.2


def test_regression_v2_keeps_cold_start_leak_visible(train_sites):
    days = _inject_leak(_clean_site(61, days=3), from_day=0, rate=3.5)
    v2 = build("leak_safe", **M3, **V2).fit(train_sites)
    p = v2.predict_sequence(days)
    assert all(x[120:].mean() > 0.8 for x in p)
    assert all(v2.components_[d.episode_id]["status"] != "METER_SHIFT" for d in days)


def test_v2_defaults_off_reproduce_v1(train_sites):
    """The v2 options are opt-in, so recorded v1 runs keep reproducing."""
    days = _inject_leak(_clean_site(62, days=3), from_day=1, rate=5.0)
    a = build("leak_safe", **M3).fit(train_sites).predict_sequence(days)
    b = build("leak_safe", **M3, raw_k_uncertain=None,
              meter_shift_requires_site_pressure=False).fit(train_sites).predict_sequence(days)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_v2_meter_swap_still_rebaselined_with_site_history(train_sites):
    days = _meter_step(_clean_site(41, days=5), from_day=2, step=2.5)
    det = build("leak_safe", **M3, **V2).fit(train_sites)
    preds = det.predict_sequence(days)
    assert det.components_[days[2].episode_id]["status"] == "METER_SHIFT"
    assert preds[3].sum() == 0 and preds[4].sum() == 0
