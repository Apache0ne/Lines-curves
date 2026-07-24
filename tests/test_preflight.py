from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from lines_curves.model import TEEDCurves
from scripts.preflight import run_preflight


def _write_config(path: Path, output: Path, natural: Path | None, fractions: tuple[float, float, float]) -> None:
    fake = path.parent / "fake_teed.pth"
    model = TEEDCurves()
    torch.save(
        {
            name: tensor
            for name, tensor in model.state_dict().items()
            if not name.startswith(model.CURVE_MODULE_PREFIXES)
        },
        fake,
    )
    config = {
        "common": {
            "seed": 1,
            "deterministic": True,
            "image_size": 32,
            "workers": 0,
            "amp": False,
            "grad_clip": 1.0,
            "use_curve_context": True,
            "validation_thresholds": [0.25, 0.35, 0.5, 0.65],
            "cache_root": str(path.parent / "cache"),
            "natural_root": str(natural) if natural else None,
            "curveml_root": str(path.parent / "curveml"),
            "output_root": str(output),
            "teed_checkpoint": str(fake),
        },
        "loss": {
            "edge": 1.0,
            "curve": 1.0,
            "dice": 0.35,
            "cldice": 0.2,
            "consistency": 0.15,
            "side": 0.2,
        },
    }
    for stage, fraction in enumerate(fractions, 1):
        config[f"stage{stage}"] = {
            "epochs": 1,
            "samples_per_epoch": 2,
            "synthetic_fraction": fraction,
            "batch_size": 2,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "initial_checkpoint": None,
        }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _write_natural(root: Path) -> None:
    for split in ("train", "val"):
        for folder in ("images", "edges", "curves"):
            (root / split / folder).mkdir(parents=True, exist_ok=True)
        image = np.zeros((32, 32, 3), np.uint8)
        edge = np.zeros((32, 32), np.uint8)
        cv2.ellipse(image, (16, 16), (10, 6), 0, 0, 360, (255, 255, 255), 1)
        cv2.ellipse(edge, (16, 16), (10, 6), 0, 0, 360, 255, 1)
        cv2.imwrite(str(root / split / "images" / "sample.png"), image)
        cv2.imwrite(str(root / split / "edges" / "sample.png"), edge)
        cv2.imwrite(str(root / split / "curves" / "sample.png"), edge)


def test_preflight_passes_complete_setup(tmp_path: Path):
    natural = tmp_path / "natural"
    _write_natural(natural)
    config = tmp_path / "config.yaml"
    _write_config(config, tmp_path / "outputs", natural, (1.0, 0.5, 0.0))
    report = run_preflight(config)
    assert report["status"] == "PASS"
    assert report["data"]["natural_train_records"] == 1
    assert len(report["stage_smoke"]) == 3
    assert report["checkpoint"]["loaded_keys"] == 36


def test_preflight_rejects_missing_required_natural_data(tmp_path: Path):
    config = tmp_path / "config.yaml"
    _write_config(config, tmp_path / "outputs", None, (1.0, 0.5, 0.0))
    report = run_preflight(config)
    assert report["status"] == "FAIL"
    assert any("needs natural training records" in error for error in report["errors"])


def test_preflight_rejects_invalid_validation_thresholds(tmp_path: Path):
    config = tmp_path / "config.yaml"
    _write_config(config, tmp_path / "outputs", None, (1.0, 1.0, 1.0))
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["common"]["validation_thresholds"] = [0.0, 1.0, "bad"]
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    report = run_preflight(config)
    assert report["status"] == "FAIL"
    assert any("validation threshold" in error.lower() for error in report["errors"])
