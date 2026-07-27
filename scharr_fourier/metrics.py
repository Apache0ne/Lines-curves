from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

Array = np.ndarray


@dataclass(frozen=True)
class BoundaryMetrics:
    precision: float
    recall: float
    f1: float
    iou: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def boundary_metrics(prediction: Array, target: Array, tolerance: float = 1.5) -> BoundaryMetrics:
    pred = np.asarray(prediction, dtype=bool)
    gt = np.asarray(target, dtype=bool)
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {gt.shape}")
    if not pred.any() and not gt.any():
        return BoundaryMetrics(1.0, 1.0, 1.0, 1.0)
    if not pred.any() or not gt.any():
        return BoundaryMetrics(0.0, 0.0, 0.0, 0.0)
    d_gt = ndimage.distance_transform_edt(~gt)
    d_pred = ndimage.distance_transform_edt(~pred)
    matched_pred = pred & (d_gt <= tolerance)
    matched_gt = gt & (d_pred <= tolerance)
    precision = float(matched_pred.sum() / max(pred.sum(), 1))
    recall = float(matched_gt.sum() / max(gt.sum(), 1))
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    union = pred | gt
    intersection = pred & gt
    iou = float(intersection.sum() / max(union.sum(), 1))
    return BoundaryMetrics(precision, recall, f1, iou)


def best_f1(probability: Array, target: Array, tolerance: float = 1.5, thresholds: int = 99) -> tuple[float, float]:
    best = (0.0, 0.5)
    for threshold in np.linspace(0.01, 0.99, thresholds):
        score = boundary_metrics(probability >= threshold, target, tolerance=tolerance).f1
        if score > best[0]:
            best = (score, float(threshold))
    return best
