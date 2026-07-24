import torch

from lines_curves.model import TEEDCurves


def test_shapes_and_stage_freeze():
    model = TEEDCurves()
    output = model(torch.randn(1, 3, 96, 104))
    assert output["edge"].shape == (1, 1, 96, 104)
    assert output["curve"].shape == (1, 1, 96, 104)
    model.freeze_for_stage(1)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(name.startswith(model.CURVE_MODULE_PREFIXES) for name in trainable)
