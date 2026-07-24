import numpy as np

from lines_curves.synthetic import CurvePointBank, render_composite


def test_synthetic_targets_are_consistent():
    image, edge, curve = render_composite(np.random.default_rng(4), 128, CurvePointBank(None))
    assert image.shape == (128, 128, 3)
    assert edge.shape == curve.shape == (128, 128)
    assert np.all((curve > 0) <= (edge > 0))
    assert edge.sum() > 0 and curve.sum() > 0
