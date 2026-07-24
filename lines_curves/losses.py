from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def balanced_bce_with_logits(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.float().clamp(0, 1)
    positive = target.sum()
    negative = target.numel() - positive
    pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 50.0)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dims)
    denominator = probability.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def _soft_erode(image: torch.Tensor) -> torch.Tensor:
    if image.shape[2] < 3 or image.shape[3] < 3:
        return image
    p1 = -F.max_pool2d(-image, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-image, (1, 3), (1, 1), (0, 1))
    return torch.minimum(p1, p2)


def _soft_dilate(image: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(image, 3, 1, 1)


def _soft_open(image: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(image))


def soft_skeletonize(image: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    image = image.clamp(0, 1)
    opened = _soft_open(image)
    skeleton = F.relu(image - opened)
    for _ in range(iterations):
        image = _soft_erode(image)
        opened = _soft_open(image)
        delta = F.relu(image - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(logits: torch.Tensor, target: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    target = target.float()
    pred_skeleton = soft_skeletonize(probability, iterations)
    target_skeleton = soft_skeletonize(target, iterations)
    dims = tuple(range(1, probability.ndim))
    tprec = (pred_skeleton * target).sum(dims) / pred_skeleton.sum(dims).clamp_min(1e-6)
    tsens = (target_skeleton * probability).sum(dims) / target_skeleton.sum(dims).clamp_min(1e-6)
    return (1.0 - (2.0 * tprec * tsens) / (tprec + tsens).clamp_min(1e-6)).mean()


@dataclass(frozen=True)
class LossWeights:
    edge: float = 1.0
    curve: float = 1.0
    dice: float = 0.35
    cldice: float = 0.20
    consistency: float = 0.15
    side: float = 0.20


def compute_loss(
    outputs: dict[str, torch.Tensor | list[torch.Tensor]],
    edge_target: torch.Tensor,
    curve_target: torch.Tensor,
    stage: int,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    edge_logits = outputs["edge"]
    curve_logits = outputs["curve"]
    assert isinstance(edge_logits, torch.Tensor) and isinstance(curve_logits, torch.Tensor)

    edge_loss = edge_logits.new_zeros(())
    edge_side_loss = edge_logits.new_zeros(())
    if stage != 1:
        edge_loss = balanced_bce_with_logits(edge_logits, edge_target) + weights.dice * dice_loss(
            edge_logits, edge_target
        )
        edge_sides = outputs["edge_sides"]
        assert isinstance(edge_sides, list)
        edge_side_loss = torch.stack(
            [balanced_bce_with_logits(side, edge_target) for side in edge_sides]
        ).mean()

    curve_loss = balanced_bce_with_logits(curve_logits, curve_target)
    curve_loss = curve_loss + weights.dice * dice_loss(curve_logits, curve_target)
    curve_loss = curve_loss + weights.cldice * cldice_loss(curve_logits, curve_target)

    curve_sides = outputs["curve_sides"]
    assert isinstance(curve_sides, list)
    curve_side_loss = torch.stack(
        [balanced_bce_with_logits(side, curve_target) for side in curve_sides]
    ).mean()

    consistency = F.relu(torch.sigmoid(curve_logits) - torch.sigmoid(edge_logits)).mean()
    total = (
        weights.edge * edge_loss
        + weights.curve * curve_loss
        + weights.side * (curve_side_loss + edge_side_loss)
        + weights.consistency * consistency
    )
    metrics = {
        "loss": float(total.detach()),
        "edge_loss": float(edge_loss.detach()),
        "curve_loss": float(curve_loss.detach()),
        "side_loss": float((curve_side_loss + edge_side_loss).detach()),
        "consistency": float(consistency.detach()),
    }
    return total, metrics
