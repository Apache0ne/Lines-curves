from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from lines_curves.curve_labels import derive_curve_mask
from lines_curves.datasets import MixedCurveDataset
from lines_curves.losses import LossWeights, compute_loss
from lines_curves.model import TEEDCurves


def main() -> None:
    model = TEEDCurves()
    total = model.parameter_report()["total"]
    assert 60_000 <= total <= 80_000, total
    x = torch.randn(2, 3, 128, 160)
    outputs = model(x)
    assert outputs["edge"].shape == (2, 1, 128, 160)
    assert outputs["curve"].shape == (2, 1, 128, 160)
    model.freeze_for_stage(1)
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert names and all(name.startswith(model.CURVE_MODULE_PREFIXES) for name in names)

    dataset = MixedCurveDataset(128, 4, 1.0, 123)
    batch = dataset[0]
    one = {key: value[None] for key, value in batch.items()}
    outputs = model(one["image"])
    loss, _ = compute_loss(outputs, one["edge"], one["curve"], 1, LossWeights())
    loss.backward()
    assert torch.isfinite(loss)

    edge = np.zeros((128, 128), np.uint8)
    cv2.ellipse(edge, (64, 64), (40, 20), 0, 0, 260, 255, 1)
    curve = derive_curve_mask(edge)
    assert curve.sum() > 0
    print(f"SMOKE_TEST=PASS PARAMS={total} LOSS={float(loss.detach()):.6f}")


if __name__ == "__main__":
    main()
