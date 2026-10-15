"""Registry of candidate solutions. New approaches register here."""

from genesis.detectors.base import Detector
from genesis.detectors.cusum import CusumDetector
from genesis.detectors.isolation_forest import IsolationForestDetector
from genesis.detectors.leak_safe import LeakSafeDetector
from genesis.detectors.mass_balance import MassBalanceThreshold

REGISTRY: dict[str, type[Detector]] = {
    cls.name: cls for cls in (MassBalanceThreshold, CusumDetector, IsolationForestDetector, LeakSafeDetector)
}


def build(name: str, **params) -> Detector:
    if name not in REGISTRY:
        raise KeyError(f"unknown detector '{name}'. available: {sorted(REGISTRY)}")
    return REGISTRY[name](**params)


__all__ = ["Detector", "REGISTRY", "build"]
