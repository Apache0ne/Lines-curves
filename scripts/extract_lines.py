from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scharr_fourier import ScharrFourierConfig, detect_lines


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract lines with multiscale Scharr + Fourier fusion")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("edge", "ridge", "hybrid"), default="hybrid")
    parser.add_argument("--low", type=float, default=0.30)
    parser.add_argument("--high", type=float, default=0.52)
    parser.add_argument("--thin", action="store_true", help="Apply Zhang-Suen skeletonization to the final binary map")
    args = parser.parse_args()

    bgr = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(args.input)
    if bgr.ndim == 3 and bgr.shape[2] >= 3:
        if bgr.shape[2] == 4:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGBA)
        else:
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        image = bgr

    config = ScharrFourierConfig(
        mode=args.mode,
        low_threshold=args.low,
        high_threshold=args.high,
        return_thinned=args.thin,
    )
    result = detect_lines(image, config)
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "probability.png", np.clip(result.probability * 255.0, 0, 255).astype(np.uint8))
    _write(args.output / "binary.png", result.binary.astype(np.uint8) * 255)
    _write(args.output / "edge_response.png", np.clip(result.edge_response * 255.0, 0, 255).astype(np.uint8))
    _write(args.output / "ridge_response.png", np.clip(result.ridge_response * 255.0, 0, 255).astype(np.uint8))
    _write(args.output / "coherence.png", np.clip(result.coherence * 255.0, 0, 255).astype(np.uint8))
    _write(args.output / "anchor_response.png", np.clip(result.anchor_response * 255.0, 0, 255).astype(np.uint8))
    _write(args.output / "fusion_response.png", np.clip(result.fusion_response * 255.0, 0, 255).astype(np.uint8))
    (args.output / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
