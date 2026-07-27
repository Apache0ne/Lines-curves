from __future__ import annotations

import numpy as np

from scharr_fourier import ScharrFourierConfig, boundary_metrics, detect_lines, make_suite


def test_constant_image_is_empty() -> None:
    image = np.full((96, 96, 3), 0.5, np.float32)
    result = detect_lines(image)
    assert result.probability.shape == image.shape[:2]
    assert np.isfinite(result.probability).all()
    assert not result.binary.any()


def test_dtype_and_alpha_support() -> None:
    image = np.zeros((80, 80, 4), np.uint8)
    image[..., :3] = 255
    image[..., 3] = 255
    image[40, 10:70, :3] = 0
    result = detect_lines(image)
    assert result.binary.dtype == np.bool_
    assert result.probability.dtype == np.float32
    assert result.binary.any()


def test_deterministic() -> None:
    case = make_suite(size=96, seed=9)[0]
    a = detect_lines(case.image)
    b = detect_lines(case.image)
    np.testing.assert_array_equal(a.binary, b.binary)
    np.testing.assert_allclose(a.probability, b.probability, rtol=0, atol=0)


def test_synthetic_quality_floor() -> None:
    scores = []
    cfg = ScharrFourierConfig()
    for case in make_suite(size=128, seed=1234):
        result = detect_lines(case.image, cfg)
        scores.append(boundary_metrics(result.binary, case.target, tolerance=2.5).f1)
    assert float(np.mean(scores)) >= 0.96
    assert float(np.min(scores)) >= 0.93
