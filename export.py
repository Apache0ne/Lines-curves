from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from lines_curves.model import TEEDCurves
from lines_curves.utils import load_model_state


class ExportWrapper(nn.Module):
    def __init__(self, model: TEEDCurves) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(image)
        return output["edge"], output["curve"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/teed_curves.onnx")
    parser.add_argument("--size", type=int, default=352)
    args = parser.parse_args()
    model = TEEDCurves()
    model.load_state_dict(load_model_state(args.checkpoint), strict=True)
    wrapper = ExportWrapper(model.eval())
    dummy = torch.randn(1, 3, args.size, args.size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
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
    print(output.resolve())


if __name__ == "__main__":
    main()
