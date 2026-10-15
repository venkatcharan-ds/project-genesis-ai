import numpy as np
import pytest

from genesis.data.synthetic_leak import GeneratorConfig, dataset_fingerprint, generate_dataset, split_episodes


def test_generation_is_deterministic(small_cfg):
    a, b = generate_dataset(small_cfg, 1), generate_dataset(small_cfg, 1)
    assert dataset_fingerprint(a) == dataset_fingerprint(b)
    assert dataset_fingerprint(a) != dataset_fingerprint(generate_dataset(small_cfg, 2))


def test_labels_match_ground_truth_leak(small_cfg):
    for ep in generate_dataset(small_cfg, 1):
        np.testing.assert_array_equal(ep.label, ep.leak_rate > 0)
        if ep.meta["has_leak"]:
            start = ep.meta["leak_start"]
            assert start >= small_cfg.warmup_minutes
            assert not ep.label[:start].any() and ep.label[start:].all()
        else:
            assert not ep.label.any()


def test_leak_raises_mass_balance_residual():
    cfg = GeneratorConfig(n_episodes=40, leak_probability=1.0, burst_share=1.0, demand_spike_probability=0.0)
    for ep in generate_dataset(cfg, 0):
        s = ep.meta["leak_start"]
        # Mean residual after the leak should exceed before by roughly the leak rate.
        gap = ep.residual[s:].mean() - ep.residual[:s].mean()
        assert gap == pytest.approx(ep.meta["leak_final_rate"], abs=1.0)


def test_split_is_by_episode_and_disjoint(splits):
    ids = {name: {e.episode_id for e in eps} for name, eps in splits.items()}
    assert not (ids["train"] & ids["val"]) and not (ids["train"] & ids["test"]) and not (ids["val"] & ids["test"])
    assert sum(len(v) for v in ids.values()) == 24


def test_split_rejects_bad_fractions(small_cfg):
    with pytest.raises(ValueError):
        split_episodes(generate_dataset(small_cfg, 0), {"train": 0.5, "test": 0.4}, 0)


def test_peak_rate_reflects_leak_actually_present():
    """Regression: late-starting gradual leaks never reach their final rate in-episode."""
    cfg = GeneratorConfig(n_episodes=60, leak_probability=1.0, burst_share=0.0)
    eps = generate_dataset(cfg, 5)
    for ep in eps:
        assert ep.meta["leak_peak_rate"] == pytest.approx(ep.leak_rate.max())
        assert ep.meta["leak_peak_rate"] <= ep.meta["leak_final_rate"] + 1e-9
    assert any(ep.meta["leak_peak_rate"] < ep.meta["leak_final_rate"] for ep in eps)
