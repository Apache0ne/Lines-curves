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

from scharr_fourier import boundary_metrics, detect_lines

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _index(root: Path) -> dict[str, Path]:
    files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    result: dict[str, Path] = {}
    for path in files:
        key = path.stem
        if key in result:
            raise RuntimeError(f"Duplicate stem {key!r}: {result[key]} and {path}")
        result[key] = path
    return result


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_target(path: Path, threshold: int) -> np.ndarray:
    target = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if target is None:
        raise RuntimeError(f"Could not read {path}")
    return target >= threshold


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired image and line-map directories")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/paired_benchmark.json"))
    parser.add_argument("--tolerance", type=float, default=2.5)
    parser.add_argument("--target-threshold", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    images = _index(args.images)
    targets = _index(args.targets)
    keys = sorted(images.keys() & targets.keys())
    if args.limit > 0:
        keys = keys[: args.limit]
    if not keys:
        raise RuntimeError("No matching image/target stems were found")

    rows = []
    for key in keys:
        image = _read_rgb(images[key])
        target = _read_target(targets[key], args.target_threshold)
        result = detect_lines(image)
        if result.binary.shape != target.shape:
            target = cv2.resize(target.astype(np.uint8), (result.binary.shape[1], result.binary.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        metrics = boundary_metrics(result.binary, target, tolerance=args.tolerance)
        rows.append({"key": key, **metrics.to_dict()})
        print(json.dumps(rows[-1], sort_keys=True))

    report = {
        "count": len(rows),
        "tolerance": args.tolerance,
        "mean_precision": float(np.mean([row["precision"] for row in rows])),
        "mean_recall": float(np.mean([row["recall"] for row in rows])),
        "mean_f1": float(np.mean([row["f1"] for row in rows])),
        "min_f1": float(np.min([row["f1"] for row in rows])),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("AGGREGATE=" + json.dumps({k: v for k, v in report.items() if k != "rows"}, sort_keys=True))


if __name__ == "__main__":
    main()
