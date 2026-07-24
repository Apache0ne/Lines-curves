from pathlib import Path

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from export import ExportWrapper
from lines_curves.model import TEEDCurves


def test_onnx_export_checker_and_dynamic_shapes(tmp_path: Path):
    wrapper = ExportWrapper(TEEDCurves().eval())
    output = tmp_path / "model.onnx"
    torch.onnx.export(
        wrapper,
        torch.randn(1, 3, 32, 32),
        output,
        input_names=["image"],
        output_names=["edge_logits", "curve_logits"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "edge_logits": {0: "batch", 2: "height", 3: "width"},
            "curve_logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
        dynamo=False,
    )
    model = onnx.load(output)
    onnx.checker.check_model(model)
    assert [item.name for item in model.graph.output] == ["edge_logits", "curve_logits"]

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    for height, width in ((32, 32), (40, 48), (33, 35)):
        outputs = session.run(None, {"image": np.zeros((1, 3, height, width), np.float32)})
        assert outputs[0].shape == outputs[1].shape == (1, 1, height, width)
