from pathlib import Path

import cv2
import numpy as np
import torch

from lines_curves.model import TEEDCurves
from lines_curves.synthetic import render_composite
from lines_curves.trainer import train_stage
from lines_curves.utils import load_model_state


def _fake_teed_checkpoint(path: Path) -> None:
    model = TEEDCurves()
    state = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if not name.startswith(model.CURVE_MODULE_PREFIXES)
    }
    torch.save(state, path)


def _base_config(tmp_path: Path, natural_root: Path | None) -> dict:
    return {
        "common": {
            "seed": 7,
            "deterministic": True,
            "image_size": 32,
            "workers": 0,
            "amp": False,
            "grad_clip": 1.0,
            "use_curve_context": True,
            "cache_root": str(tmp_path / "cache"),
            "natural_root": str(natural_root) if natural_root else None,
            "curveml_root": str(tmp_path / "curveml"),
            "output_root": str(tmp_path / "outputs"),
            "teed_checkpoint": str(tmp_path / "fake_teed.pth"),
        },
        "stage1": {
            "epochs": 1,
            "samples_per_epoch": 2,
            "synthetic_fraction": 1.0,
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 2e-4,
            "initial_checkpoint": None,
        },
        "stage2": {
            "epochs": 1,
            "samples_per_epoch": 2,
            "synthetic_fraction": 0.5,
            "batch_size": 4,
            "lr": 3.5e-4,
            "weight_decay": 2e-4,
            "initial_checkpoint": None,
        },
        "stage3": {
            "epochs": 1,
            "samples_per_epoch": 2,
            "synthetic_fraction": 0.0,
            "batch_size": 4,
            "lr": 5e-5,
            "weight_decay": 1e-4,
            "initial_checkpoint": None,
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


def _write_natural(root: Path) -> None:
    for split, seed in (("train", 10), ("val", 20)):
        for folder in ("images", "edges", "curves"):
            (root / split / folder).mkdir(parents=True, exist_ok=True)
        image, edge, curve = render_composite(
            np.random.default_rng(seed), 32, min_curves=1, max_curves=2
        )
        cv2.imwrite(str(root / split / "images" / "sample.png"), image)
        cv2.imwrite(str(root / split / "edges" / "sample.png"), edge)
        cv2.imwrite(str(root / split / "curves" / "sample.png"), curve)


def test_stage1_without_validation_still_saves_best(tmp_path: Path):
    _fake_teed_checkpoint(tmp_path / "fake_teed.pth")
    config = _base_config(tmp_path, None)
    checkpoint = train_stage(config, 1)
    assert checkpoint.exists()
    assert checkpoint.name == "best.pt"
    assert checkpoint.with_name("best.safetensors").exists()
    assert checkpoint.with_name("best_fp16.safetensors").exists()


def test_three_stage_checkpoint_handoff(tmp_path: Path):
    _fake_teed_checkpoint(tmp_path / "fake_teed.pth")
    natural_root = tmp_path / "natural"
    _write_natural(natural_root)
    config = _base_config(tmp_path, natural_root)

    for stage in (1, 2, 3):
        checkpoint = train_stage(config, stage)
        assert checkpoint.exists()

    final_state = load_model_state(tmp_path / "outputs" / "stage3" / "best.safetensors")
    model = TEEDCurves()
    model.load_state_dict(final_state, strict=True)
    output = model(torch.randn(1, 3, 32, 32))
    assert output["edge"].shape == output["curve"].shape == (1, 1, 32, 32)


def test_exact_resume_matches_uninterrupted_training(tmp_path: Path):
    import copy
    import shutil

    _fake_teed_checkpoint(tmp_path / "fake_teed.pth")
    natural_root = tmp_path / "natural"
    _write_natural(natural_root)

    full_config = _base_config(tmp_path, natural_root)
    full_config["common"]["output_root"] = str(tmp_path / "full")
    full_config["stage1"]["epochs"] = 2
    train_stage(full_config, 1)

    resume_root = tmp_path / "resumed"
    resume_stage = resume_root / "stage1"
    resume_stage.mkdir(parents=True)
    interrupted = tmp_path / "full" / "stage1" / "epoch_000.pt"
    resume_checkpoint = resume_stage / "last.pt"
    shutil.copy2(interrupted, resume_checkpoint)

    resumed_config = copy.deepcopy(full_config)
    resumed_config["common"]["output_root"] = str(resume_root)
    train_stage(resumed_config, 1, resume=resume_checkpoint)

    uninterrupted = torch.load(
        tmp_path / "full" / "stage1" / "last.pt", map_location="cpu", weights_only=False
    )
    resumed = torch.load(
        resume_root / "stage1" / "last.pt", map_location="cpu", weights_only=False
    )
    assert uninterrupted["epoch"] == resumed["epoch"] == 1
    assert uninterrupted["global_step"] == resumed["global_step"]
    for name, tensor in uninterrupted["model"].items():
        assert torch.equal(tensor, resumed["model"][name]), name


def test_completed_checkpoint_copied_to_new_output_exports_best(tmp_path: Path):
    import copy
    import shutil

    _fake_teed_checkpoint(tmp_path / "fake_teed.pth")
    config = _base_config(tmp_path, None)
    config["common"]["output_root"] = str(tmp_path / "source")
    train_stage(config, 1)

    copied = tmp_path / "copied_last.pt"
    shutil.copy2(tmp_path / "source" / "stage1" / "last.pt", copied)
    restored_config = copy.deepcopy(config)
    restored_config["common"]["output_root"] = str(tmp_path / "restored")
    best = train_stage(restored_config, 1, resume=copied)

    assert best.exists()
    assert best.with_name("best.safetensors").exists()
    assert best.with_name("best_fp16.safetensors").exists()
    source = torch.load(copied, map_location="cpu", weights_only=False)["model"]
    restored = torch.load(best, map_location="cpu", weights_only=False)["model"]
    for name, tensor in source.items():
        assert torch.equal(tensor, restored[name]), name
