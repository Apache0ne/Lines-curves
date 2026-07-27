from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time
from typing import Callable

import cv2
import numpy as np
from scipy import ndimage

from scharr_fourier import detect_lines

Array = np.ndarray
Inverse = Callable[[Array], Array]
TRANSFORMS = (
    "repeat",
    "flip_h",
    "flip_v",
    "rot90",
    "exposure",
    "jpeg70",
    "noise2",
    "resize75",
)


def _resize_long_side(image: Array, maximum: int) -> Array:
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1.0:
        return image
    return cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _transform(image: Array, name: str, seed: int) -> tuple[Array, Inverse]:
    if name == "repeat":
        return image.copy(), lambda value: value
    if name == "flip_h":
        return np.ascontiguousarray(image[:, ::-1]), lambda value: np.ascontiguousarray(value[:, ::-1])
    if name == "flip_v":
        return np.ascontiguousarray(image[::-1, :]), lambda value: np.ascontiguousarray(value[::-1, :])
    if name == "rot90":
        return np.ascontiguousarray(np.rot90(image)), lambda value: np.ascontiguousarray(np.rot90(value, -1))
    if name == "exposure":
        value = image.astype(np.float32) / 255.0
        value = np.clip((value**0.78) * 1.08 + 0.015, 0.0, 1.0)
        return (value * 255.0 + 0.5).astype(np.uint8), lambda output: output
    if name == "jpeg70":
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), lambda output: output
    if name == "noise2":
        rng = np.random.default_rng(seed)
        value = np.clip(
            image.astype(np.float32) + rng.normal(0.0, 2.0, image.shape),
            0.0,
            255.0,
        ).astype(np.uint8)
        return value, lambda output: output
    if name == "resize75":
        height, width = image.shape[:2]
        small = cv2.resize(
            image,
            (max(8, round(width * 0.75)), max(8, round(height * 0.75))),
            interpolation=cv2.INTER_AREA,
        )
        restored = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        return restored, lambda output: output
    raise ValueError(f"Unsupported transform: {name}")


def _boundary_score(reference: Array, prediction: Array, tolerance: float) -> dict[str, float]:
    reference = np.asarray(reference, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if not reference.any() and not prediction.any():
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not reference.any() or not prediction.any():
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    distance_reference = ndimage.distance_transform_edt(~reference)
    distance_prediction = ndimage.distance_transform_edt(~prediction)
    precision = float((distance_reference[prediction] <= tolerance).mean())
    recall = float((distance_prediction[reference] <= tolerance).mean())
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def _correlation(first: Array, second: Array) -> float:
    a = first.astype(np.float64).ravel()
    b = second.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float(np.array_equal(first, second))


def _process(path: str, maximum: int, tolerance: float, output_root: str | None) -> dict[str, object]:
    cv2.setNumThreads(1)
    image_path = Path(path)
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read {image_path}")
    image = _resize_long_side(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), maximum)
    started = time.perf_counter()
    baseline = detect_lines(image)
    base_seconds = time.perf_counter() - started

    if output_root:
        directory = Path(output_root) / image_path.stem
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / "probability.png"), np.clip(baseline.probability * 255.0, 0, 255).astype(np.uint8))
        cv2.imwrite(str(directory / "binary.png"), baseline.binary.astype(np.uint8) * 255)
        overlay = image.copy()
        overlay[baseline.binary] = (255, 32, 32)
        cv2.imwrite(str(directory / "overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    seed = int.from_bytes(image_path.stem.encode("utf-8"), "little", signed=False) % (2**32)
    rows: list[dict[str, object]] = []
    for transform_name in TRANSFORMS:
        transformed, inverse = _transform(image, transform_name, seed)
        started = time.perf_counter()
        result = detect_lines(transformed)
        elapsed = time.perf_counter() - started
        binary = inverse(result.binary)
        probability = inverse(result.probability)
        row = {
            "transform": transform_name,
            **_boundary_score(baseline.binary, binary, tolerance),
            "probability_correlation": _correlation(baseline.probability, probability),
            "seconds": elapsed,
        }
        rows.append(row)
    return {
        "image": image_path.name,
        "shape": list(image.shape),
        "base_seconds": base_seconds,
        "line_density": float(baseline.binary.mean()),
        "transforms": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-free CPU robustness benchmark for natural images")
    parser.add_argument("--images", required=True, help="Directory containing images")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--maps", help="Optional directory for probability, binary, and overlay images")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff")
    paths = sorted({path for pattern in patterns for path in Path(args.images).glob(pattern)})
    if not paths:
        raise SystemExit("No images found")

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_process, str(path), args.max_side, args.tolerance, args.maps): path
            for path in paths
        }
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results.append(row)
            print(f"[{index}/{len(paths)}] {row['image']}", flush=True)
    results.sort(key=lambda row: str(row["image"]))

    transforms: dict[str, dict[str, float]] = {}
    for name in TRANSFORMS:
        values = [
            next(row for row in image["transforms"] if row["transform"] == name)
            for image in results
        ]
        transforms[name] = {
            key: float(np.mean([float(value[key]) for value in values]))
            for key in ("precision", "recall", "f1", "probability_correlation", "seconds")
        }
        transforms[name]["min_f1"] = float(min(float(value["f1"]) for value in values))

    nonrepeat = [
        float(row["f1"])
        for image in results
        for row in image["transforms"]
        if row["transform"] != "repeat"
    ]
    report = {
        "benchmark_type": "label-free transformation consistency; not ground-truth line accuracy",
        "environment": {
            "cpu_count": os.cpu_count(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "max_side": args.max_side,
            "workers": args.workers,
            "tolerance": args.tolerance,
        },
        "aggregate": {
            "images": len(results),
            "wall_seconds": time.perf_counter() - started,
            "mean_base_seconds": float(np.mean([float(row["base_seconds"]) for row in results])),
            "mean_line_density": float(np.mean([float(row["line_density"]) for row in results])),
            "mean_nonrepeat_f1": float(np.mean(nonrepeat)),
            "min_nonrepeat_f1": float(min(nonrepeat)),
            "perfect_transform_image_pairs": int(sum(value == 1.0 for value in nonrepeat)),
            "total_transform_image_pairs": len(nonrepeat),
        },
        "transforms": transforms,
        "images": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"aggregate": report["aggregate"], "transforms": transforms}, indent=2))


if __name__ == "__main__":
    main()
