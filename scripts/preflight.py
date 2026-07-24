from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from lines_curves.datasets import MixedCurveDataset, discover_natural_records
from lines_curves.losses import LossWeights, compute_loss
from lines_curves.model import TEEDCurves
from lines_curves.synthetic import CurvePointBank
from lines_curves.utils import load_yaml


def _positive_int(value: Any, name: str, errors: list[str]) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be an integer, got {value!r}")
        return 0
    if parsed <= 0:
        errors.append(f"{name} must be greater than zero, got {parsed}")
    return parsed


def _load_teed_checkpoint(model: TEEDCurves, common: dict[str, Any], download: bool) -> dict[str, Any] | None:
    configured = common.get("teed_checkpoint")
    checkpoint: Path | None = None
    if configured:
        checkpoint = Path(configured).expanduser().resolve()
        if not checkpoint.exists():
            raise FileNotFoundError(f"Configured TEED checkpoint does not exist: {checkpoint}")
    elif download:
        from huggingface_hub import hf_hub_download

        cache_root = Path(common.get("cache_root", ROOT / ".cache")).expanduser().resolve() / "teed"
        checkpoint = Path(
            hf_hub_download(repo_id="fal/teed", filename="5_model.pth", local_dir=cache_root)
        )
    if checkpoint is None:
        return None
    report = model.load_teed_checkpoint(checkpoint)
    if int(report["loaded_keys"]) < 20:
        raise RuntimeError(f"Too few compatible TEED tensors loaded from {checkpoint}: {report}")
    return {"path": str(checkpoint), **report}


def run_preflight(config_path: str | Path, download_teed: bool = False) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = load_yaml(config_path)
    errors: list[str] = []
    warnings: list[str] = []
    common = config.get("common")
    if not isinstance(common, dict):
        return {"status": "FAIL", "errors": ["Missing mapping: common"], "warnings": []}

    image_size = _positive_int(common.get("image_size"), "common.image_size", errors)
    workers = common.get("workers", 0)
    try:
        workers = int(workers)
        if workers < 0:
            errors.append("common.workers must be zero or greater")
    except (TypeError, ValueError):
        errors.append(f"common.workers must be an integer, got {workers!r}")
        workers = 0

    stage_rows: list[dict[str, Any]] = []
    for stage in (1, 2, 3):
        key = f"stage{stage}"
        stage_cfg = config.get(key)
        if not isinstance(stage_cfg, dict):
            errors.append(f"Missing mapping: {key}")
            continue
        epochs = _positive_int(stage_cfg.get("epochs"), f"{key}.epochs", errors)
        samples = _positive_int(stage_cfg.get("samples_per_epoch"), f"{key}.samples_per_epoch", errors)
        batch = _positive_int(stage_cfg.get("batch_size"), f"{key}.batch_size", errors)
        try:
            fraction = float(stage_cfg.get("synthetic_fraction"))
            if not 0.0 <= fraction <= 1.0:
                errors.append(f"{key}.synthetic_fraction must be within [0, 1]")
        except (TypeError, ValueError):
            fraction = -1.0
            errors.append(f"{key}.synthetic_fraction must be numeric")
        try:
            lr = float(stage_cfg.get("lr"))
            if lr <= 0:
                errors.append(f"{key}.lr must be greater than zero")
        except (TypeError, ValueError):
            lr = 0.0
            errors.append(f"{key}.lr must be numeric")
        stage_rows.append(
            {
                "stage": stage,
                "epochs": epochs,
                "samples_per_epoch": samples,
                "batch_size": batch,
                "synthetic_fraction": fraction,
                "lr": lr,
            }
        )

    natural_root_raw = common.get("natural_root")
    natural_root = Path(natural_root_raw).expanduser().resolve() if natural_root_raw else None
    train_records = discover_natural_records(natural_root, "train")
    val_records = discover_natural_records(natural_root, "val")
    for row in stage_rows:
        if row["synthetic_fraction"] < 1.0 and not train_records:
            errors.append(
                f"stage{row['stage']} needs natural training records because "
                f"synthetic_fraction={row['synthetic_fraction']}, but none were found under {natural_root}"
            )
    if not val_records:
        warnings.append("No natural validation records found; best checkpoints will use training loss fallback.")

    curveml_root_raw = common.get("curveml_root")
    curveml_root = Path(curveml_root_raw).expanduser().resolve() if curveml_root_raw else None
    point_bank = CurvePointBank(curveml_root)
    if len(point_bank) == 0:
        warnings.append("No CurveML point sets found; exact procedural curve generation will be used.")

    model = TEEDCurves(use_curve_context=bool(common.get("use_curve_context", True)))
    parameter_report = model.parameter_report()
    if parameter_report["total"] != 67_980:
        errors.append(f"Unexpected model parameter count: {parameter_report['total']} (expected 67980)")

    checkpoint_report = None
    try:
        checkpoint_report = _load_teed_checkpoint(model, common, download_teed)
    except Exception as exc:
        errors.append(f"TEED checkpoint validation failed: {exc}")
    if checkpoint_report is None:
        warnings.append("Public TEED checkpoint was not downloaded during preflight; Stage 1 will download it.")

    loss_weights = LossWeights(**config.get("loss", {}))
    smoke_size = min(max(image_size, 16), 96) if image_size else 32
    stage_smoke: list[dict[str, Any]] = []
    if not errors:
        for row in stage_rows:
            stage = int(row["stage"])
            try:
                dataset = MixedCurveDataset(
                    size=smoke_size,
                    samples_per_epoch=1,
                    synthetic_fraction=float(row["synthetic_fraction"]),
                    seed=int(common.get("seed", 1021)) + stage * 100,
                    natural_root=natural_root,
                    split="train",
                    curveml_root=curveml_root,
                    augment=True,
                )
                sample = dataset[0]
                model.freeze_for_stage(stage)
                model.zero_grad(set_to_none=True)
                outputs = model(sample["image"].unsqueeze(0))
                loss, pieces = compute_loss(
                    outputs,
                    sample["edge"].unsqueeze(0),
                    sample["curve"].unsqueeze(0),
                    stage,
                    loss_weights,
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss: {loss.item()}")
                loss.backward()
                finite_gradients = all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )
                if not finite_gradients:
                    raise FloatingPointError("non-finite gradients")
                stage_smoke.append(
                    {
                        "stage": stage,
                        "loss": float(loss.detach()),
                        "trainable_parameters": model.parameter_report()["trainable"],
                        "pieces": pieces,
                    }
                )
            except Exception as exc:
                errors.append(f"Stage {stage} smoke failed: {exc}")

    output_root_raw = common.get("output_root")
    output_root = Path(output_root_raw).expanduser().resolve() if output_root_raw else None
    return {
        "status": "PASS" if not errors else "FAIL",
        "config": str(config_path),
        "errors": errors,
        "warnings": warnings,
        "paths": {
            "natural_root": str(natural_root) if natural_root else None,
            "curveml_root": str(curveml_root) if curveml_root else None,
            "output_root": str(output_root) if output_root else None,
        },
        "data": {
            "natural_train_records": len(train_records),
            "natural_val_records": len(val_records),
            "curveml_point_sets": len(point_bank),
        },
        "model": parameter_report,
        "checkpoint": checkpoint_report,
        "stages": stage_rows,
        "stage_smoke": stage_smoke,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TEED-Curves config, paths, data, and training smoke")
    parser.add_argument("--config", default="configs/colab.yaml")
    parser.add_argument("--download-teed", action="store_true")
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    report = run_preflight(args.config, download_teed=args.download_teed)
    if args.report:
        report_path = Path(args.report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"PREFLIGHT_REPORT={report_path}")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"PREFLIGHT_STATUS={report['status']}")
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
