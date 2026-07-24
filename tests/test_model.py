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
    assert sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) == 9070


def test_teed_compatible_parameter_layout_and_head_copy():
    source = TEEDCurves()
    teed_state = {
        name: tensor.clone()
        for name, tensor in source.state_dict().items()
        if not name.startswith(source.CURVE_MODULE_PREFIXES)
    }
    assert len(teed_state) == 36
    assert sum(tensor.numel() for tensor in teed_state.values()) == 58910

    target = TEEDCurves()
    report = target.load_teed_checkpoint(teed_state)
    assert report["loaded_keys"] == 36
    for edge_module, curve_module in (
        (target.up_block_1, target.curve_up_block_1),
        (target.up_block_2, target.curve_up_block_2),
        (target.up_block_3, target.curve_up_block_3),
        (target.block_cat, target.curve_block_cat),
    ):
        for edge_tensor, curve_tensor in zip(
            edge_module.state_dict().values(), curve_module.state_dict().values(), strict=True
        ):
            assert torch.equal(edge_tensor, curve_tensor)
