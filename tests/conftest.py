import pytest

from genesis.data.synthetic_leak import GeneratorConfig, generate_dataset, split_episodes


@pytest.fixture(scope="session")
def small_cfg() -> GeneratorConfig:
    return GeneratorConfig(n_episodes=24, episode_minutes=720, warmup_minutes=60)


@pytest.fixture(scope="session")
def splits(small_cfg):
    return split_episodes(generate_dataset(small_cfg, seed=7), {"train": 0.5, "val": 0.25, "test": 0.25}, seed=7)


def tiny_experiment_config() -> dict:
    """Small but complete config for fast end-to-end tests."""
    return {
        "experiment": {"name": "test_run", "synthetic": True, "seed": 3},
        "problem": {"statement": "test problem"},
        "data": {"n_episodes": 16, "episode_minutes": 600, "warmup_minutes": 60},
        "split": {"train": 0.5, "val": 0.25, "test": 0.25},
        "candidates": [
            {"name": "mass_balance_threshold", "params": {"window": 15, "k": 4.0},
             "search_space": {"window": [5, 15, 30], "k": [3.0, 4.0, 5.0]}},
            {"name": "cusum", "params": {"drift": 0.5, "h": 20.0},
             "search_space": {"drift": [0.25, 0.5, 1.0], "h": [10.0, 20.0, 40.0]}},
        ],
        "evaluation": {"bootstrap_samples": 50},
        "improvement": {"max_iterations": 3, "max_evaluations": 6, "min_delta": 0.001,
                        "max_fpr": 0.01, "fpr_penalty": 10.0},
    }
