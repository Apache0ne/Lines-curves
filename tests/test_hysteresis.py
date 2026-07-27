from __future__ import annotations

import time

import numpy as np
from scipy import ndimage

from scharr_fourier.core import _hysteresis


def _iterative_hysteresis(probability: np.ndarray, low: float, high: float) -> np.ndarray:
    weak = probability >= low
    current = probability >= high
    structure = np.ones((3, 3), dtype=bool)
    while True:
        grown = weak & ndimage.binary_dilation(current, structure=structure)
        if np.array_equal(grown, current):
            return current
        current = grown


def test_connected_component_hysteresis_matches_iterative_reference() -> None:
    rng = np.random.default_rng(1234)
    for shape in ((9, 11), (32, 25), (73, 81)):
        for _ in range(100):
            probability = rng.random(shape, dtype=np.float32)
            low = float(rng.uniform(0.0, 0.6))
            high = float(rng.uniform(low, 1.0))
            expected = _iterative_hysteresis(probability, low, high)
            actual = _hysteresis(probability, low, high)
            np.testing.assert_array_equal(actual, expected)


def test_connected_component_hysteresis_handles_long_propagation() -> None:
    probability = np.full((256, 256), 0.4, dtype=np.float32)
    probability[0, 0] = 1.0
    started = time.perf_counter()
    result = _hysteresis(probability, 0.3, 0.8)
    elapsed = time.perf_counter() - started
    assert result.all()
    assert elapsed < 0.1
