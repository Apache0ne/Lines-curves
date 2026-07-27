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

from scharr_fourier import (
    ScharrFourierConfig,
    best_f1,
    boundary_metrics,
    canny_binary,
    detect_lines,
    fourier_scharr_probability,
    make_suite,
    scharr_probability,
    sobel_probability,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Scharr + Fourier line extraction")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tolerance", type=float, default=2.5)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    config = ScharrFourierConfig()
    rows = []
    method_scores: dict[str, list[float]] = {
        "sobel_ods": [],
        "scharr_ods": [],
        "fourier_scharr_ods": [],
        "canny": [],
        "scharr_fourier_fusion": [],
        "scharr_fourier_fusion_ods": [],
    }

    for case in make_suite(size=args.size, seed=args.seed):
        result = detect_lines(case.image, config)
        default = boundary_metrics(result.binary, case.target, tolerance=args.tolerance)
        ods_f1, ods_threshold = best_f1(result.probability, case.target, tolerance=args.tolerance)
        sobel_f1, _ = best_f1(sobel_probability(case.image), case.target, tolerance=args.tolerance)
        scharr_f1, _ = best_f1(scharr_probability(case.image), case.target, tolerance=args.tolerance)
        fs_f1, _ = best_f1(fourier_scharr_probability(case.image), case.target, tolerance=args.tolerance)
        canny_f1 = boundary_metrics(canny_binary(case.image), case.target, tolerance=args.tolerance).f1

        method_scores["sobel_ods"].append(sobel_f1)
        method_scores["scharr_ods"].append(scharr_f1)
        method_scores["fourier_scharr_ods"].append(fs_f1)
        method_scores["canny"].append(canny_f1)
        method_scores["scharr_fourier_fusion"].append(default.f1)
        method_scores["scharr_fourier_fusion_ods"].append(ods_f1)

        row = {
            "case": case.name,
            **default.to_dict(),
            "ods_f1": ods_f1,
            "ods_threshold": ods_threshold,
            "baselines": {
                "sobel_ods_f1": sobel_f1,
                "scharr_ods_f1": scharr_f1,
                "fourier_scharr_ods_f1": fs_f1,
                "canny_f1": canny_f1,
            },
        }
        rows.append(row)
        case_dir = args.output / case.name
        case_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(case_dir / "input.png"), np.clip(case.image[..., ::-1] * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(str(case_dir / "target.png"), case.target.astype(np.uint8) * 255)
        cv2.imwrite(str(case_dir / "probability.png"), np.clip(result.probability * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(str(case_dir / "binary.png"), result.binary.astype(np.uint8) * 255)
        print(json.dumps(row, sort_keys=True))

    comparison = {
        name: {"mean_f1": float(np.mean(values)), "min_f1": float(np.min(values))}
        for name, values in method_scores.items()
    }
    aggregate = {
        "mean_precision": float(np.mean([r["precision"] for r in rows])),
        "mean_recall": float(np.mean([r["recall"] for r in rows])),
        "mean_f1": float(np.mean([r["f1"] for r in rows])),
        "mean_ods_f1": float(np.mean([r["ods_f1"] for r in rows])),
        "min_f1": float(np.min([r["f1"] for r in rows])),
        "cases": len(rows),
        "size": args.size,
        "tolerance": args.tolerance,
        "seed": args.seed,
        "config": config.to_dict(),
    }
    report = {"aggregate": aggregate, "comparison": comparison, "cases": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("AGGREGATE=" + json.dumps(aggregate, sort_keys=True))
    print("COMPARISON=" + json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
