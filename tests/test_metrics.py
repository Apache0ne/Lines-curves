from __future__ import annotations

import numpy as np

from scharr_fourier import boundary_metrics


def test_boundary_metrics_exact() -> None:
    target = np.zeros((32, 32), dtype=bool)
    target[16, 4:28] = True
    metrics = boundary_metrics(target, target, tolerance=0.0)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.iou == 1.0


def test_boundary_metrics_tolerance() -> None:
    target = np.zeros((32, 32), dtype=bool)
    prediction = np.zeros_like(target)
    target[16, 4:28] = True
    prediction[17, 4:28] = True
    assert boundary_metrics(prediction, target, tolerance=1.0).f1 == 1.0
    assert boundary_metrics(prediction, target, tolerance=0.0).f1 == 0.0
