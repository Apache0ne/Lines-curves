from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from lines_curves.model import TEEDCurves
from lines_curves.utils import load_model_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/inference")
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TEEDCurves()
    model.load_state_dict(load_model_state(args.checkpoint), strict=True)
    model.eval().to(device)

    image = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.input)
    original_hw = image.shape[:2]
    h = ((image.shape[0] + 7) // 8) * 8
    w = ((image.shape[1] + 7) // 8) * 8
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_CUBIC)
    mean = np.asarray([103.939, 116.779, 123.68], np.float32)
    tensor = torch.from_numpy((resized.astype(np.float32) - mean).transpose(2, 0, 1))[None]
    with torch.inference_mode():
        outputs = model(tensor.to(device))
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("edge", "curve"):
        probability = torch.sigmoid(outputs[name])[0, 0].cpu().numpy()
        probability = cv2.resize(probability, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(output_dir / f"{name}_probability.png"), np.clip(probability * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(str(output_dir / f"{name}_binary.png"), (probability >= args.threshold).astype(np.uint8) * 255)


if __name__ == "__main__":
    main()
