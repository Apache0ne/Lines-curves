import torch

from lines_curves.losses import LossWeights, compute_loss
from lines_curves.model import TEEDCurves


def test_all_stages_have_finite_gradients_for_extreme_targets():
    for stage in (1, 2, 3):
        for target_value in (0.0, 1.0):
            model = TEEDCurves()
            model.freeze_for_stage(stage)
            outputs = model(torch.randn(2, 3, 33, 35))
            edge = torch.full((2, 1, 33, 35), target_value)
            curve = torch.full((2, 1, 33, 35), target_value)
            loss, _ = compute_loss(outputs, edge, curve, stage, LossWeights())
            loss.backward()
            assert torch.isfinite(loss)
            assert all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
