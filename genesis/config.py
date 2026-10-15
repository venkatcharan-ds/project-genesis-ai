"""Experiment config loading and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_KEYS = ("experiment", "problem", "data", "split", "candidates", "improvement", "evaluation")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config {path} missing keys: {missing}")
    if cfg["experiment"].get("synthetic") is not True:
        # The MVP only supports synthetic data; real-data runs need a separate loader.
        raise ValueError("only synthetic experiments are supported in this version")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
