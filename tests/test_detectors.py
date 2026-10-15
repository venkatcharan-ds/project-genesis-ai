import numpy as np
import pytest

from genesis.data.synthetic_leak import Episode
from genesis.detectors import REGISTRY, build
from genesis.detectors.base import causal_moving_average


def test_causal_moving_average_uses_only_past():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(causal_moving_average(x, 2), [1.0, 1.5, 2.5, 3.5])
    # Changing a future value must not change earlier outputs.
    y = x.copy(); y[-1] = 100.0
    np.testing.assert_allclose(causal_moving_average(y, 2)[:3], causal_moving_average(x, 2)[:3])


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_detector_output_shape_and_type(name, splits):
    det = build(name).fit(splits["train"])
    for ep in splits["test"]:
        p = det.predict(ep)
        assert p.dtype == bool and p.shape == ep.label.shape


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_detector_is_causal(name, splits):
    """Perturbing the future must not change past alarms."""
    det = build(name).fit(splits["train"])
    ep = splits["test"][0]
    cut = len(ep) // 2
    future = Episode(ep.episode_id, ep.flow_in.copy(), ep.flow_out.copy(), ep.pressure.copy(),
                     ep.leak_rate, ep.label, ep.meta)
    future.flow_in[cut:] += 50.0
    np.testing.assert_array_equal(det.predict(ep)[:cut], det.predict(future)[:cut])


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_detector_catches_obvious_burst(name, splits):
    det = build(name).fit(splits["train"])
    ep = splits["test"][0]
    n = len(ep)
    leak = np.zeros(n); leak[n // 2:] = 15.0   # 15% of base demand: unmistakable
    burst = Episode(-1, ep.flow_in + leak, ep.flow_out, ep.pressure - 0.075 * (leak > 0),
                    leak, leak > 0, ep.meta)
    assert det.predict(burst)[n // 2:].mean() > 0.8


def test_unknown_params_and_names_rejected():
    with pytest.raises(ValueError):
        build("cusum", bogus=1)
    with pytest.raises(KeyError):
        build("does_not_exist")
