from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from safetensors.torch import save_file

from .datasets import MixedCurveDataset, NaturalValidationDataset
from .losses import LossWeights, compute_loss
from .model import TEEDCurves
from .utils import append_jsonl, atomic_torch_save, binary_metrics, seed_everything


def _download_teed_checkpoint(cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="fal/teed", filename="5_model.pth", local_dir=cache_dir)
    return Path(path)


def _amp_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled)
    return contextlib.nullcontext()


def _make_loader(dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=shuffle,
    )


def _checkpoint_payload(
    model: TEEDCurves,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    stage: int,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    best_f1: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "stage": stage,
        "epoch": epoch,
        "global_step": global_step,
        "best_f1": best_f1,
        "config": config,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _save_lightweight_weights(model: TEEDCurves, output_dir: Path, stem: str = "best") -> None:
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state, output_dir / f"{stem}.safetensors", metadata={"architecture": "TEEDCurves"})
    fp16_state = {
        name: tensor.half() if tensor.is_floating_point() else tensor
        for name, tensor in state.items()
    }
    save_file(
        fp16_state,
        output_dir / f"{stem}_fp16.safetensors",
        metadata={"architecture": "TEEDCurves", "precision": "fp16"},
    )


def train_stage(config: dict[str, Any], stage: int, resume: str | Path | None = None) -> Path:
    if stage not in (1, 2, 3):
        raise ValueError("stage must be 1, 2, or 3")
    common = config["common"]
    stage_cfg = config[f"stage{stage}"]
    seed_everything(int(common["seed"]), bool(common.get("deterministic", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(common["output_root"]).expanduser().resolve() / f"stage{stage}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = TEEDCurves(use_curve_context=bool(common.get("use_curve_context", True)))
    initialization = stage_cfg.get("initial_checkpoint")
    if resume:
        initialization = resume
    if initialization:
        payload = torch.load(initialization, map_location="cpu", weights_only=False)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        model.load_state_dict(state, strict=True)
    elif stage == 1:
        pretrained = common.get("teed_checkpoint")
        if not pretrained:
            pretrained = _download_teed_checkpoint(Path(common["cache_root"]) / "teed")
        report = model.load_teed_checkpoint(pretrained)
        if int(report["loaded_keys"]) < 20:
            raise RuntimeError(f"Too few TEED weights loaded: {report}")
    else:
        previous = Path(common["output_root"]) / f"stage{stage - 1}" / "best.pt"
        if not previous.exists():
            raise FileNotFoundError(f"Previous-stage checkpoint is missing: {previous}")
        payload = torch.load(previous, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"], strict=True)

    model.freeze_for_stage(stage)
    model.to(device)
    report = model.parameter_report()
    print(f"Stage {stage}: device={device}, parameters={report}")
    if stage == 1 and report["trainable"] >= report["total"]:
        raise RuntimeError("Stage 1 freeze failed: all parameters remain trainable")

    natural_root = common.get("natural_root")
    train_dataset = MixedCurveDataset(
        size=int(common["image_size"]),
        samples_per_epoch=int(stage_cfg["samples_per_epoch"]),
        synthetic_fraction=float(stage_cfg["synthetic_fraction"]),
        seed=int(common["seed"]) + stage * 100,
        natural_root=natural_root,
        split="train",
        curveml_root=common.get("curveml_root"),
        augment=True,
    )
    train_loader = _make_loader(
        train_dataset,
        int(stage_cfg["batch_size"]),
        int(common["workers"]),
        shuffle=True,
    )
    val_loader = None
    if natural_root and (Path(natural_root) / "val").exists():
        val_dataset = NaturalValidationDataset(natural_root, "val", int(common["image_size"]))
        val_loader = _make_loader(val_dataset, int(stage_cfg["batch_size"]), int(common["workers"]), False)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(stage_cfg["lr"]), weight_decay=float(stage_cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(stage_cfg["epochs"])), eta_min=float(stage_cfg["lr"]) * 0.05
    )
    amp_enabled = bool(common.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    weights = LossWeights(**config.get("loss", {}))
    start_epoch, global_step, best_f1 = 0, 0, -1.0

    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("stage") == stage and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
            scheduler.load_state_dict(payload["scheduler"])
            scaler.load_state_dict(payload.get("scaler", {}))
            start_epoch = int(payload["epoch"]) + 1
            global_step = int(payload.get("global_step", 0))
            best_f1 = float(payload.get("best_f1", -1.0))

    for epoch in range(start_epoch, int(stage_cfg["epochs"])):
        train_dataset.set_epoch(epoch)
        model.train()
        running = {name: 0.0 for name in ("loss", "edge_loss", "curve_loss", "side_loss", "consistency")}
        start = time.perf_counter()
        progress = tqdm(train_loader, desc=f"stage {stage} epoch {epoch + 1}")
        for batch in progress:
            image = batch["image"].to(device, non_blocking=True)
            edge = batch["edge"].to(device, non_blocking=True)
            curve = batch["curve"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp_context(device, amp_enabled):
                outputs = model(image)
                loss, pieces = compute_loss(outputs, edge, curve, stage, weights)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at stage={stage}, epoch={epoch}, step={global_step}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(trainable, float(common["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            for name, value in pieces.items():
                running[name] += value
            progress.set_postfix(loss=f"{pieces['loss']:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()
        batches = max(len(train_loader), 1)
        epoch_row = {
            "stage": stage,
            "epoch": epoch,
            "global_step": global_step,
            "seconds": time.perf_counter() - start,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value / batches for name, value in running.items()},
        }

        validation_f1 = -epoch_row["train_loss"]
        if val_loader is not None:
            model.eval()
            aggregate = {"edge_f1": 0.0, "curve_f1": 0.0}
            with torch.no_grad():
                for batch in val_loader:
                    image = batch["image"].to(device, non_blocking=True)
                    outputs = model(image)
                    edge_metrics = binary_metrics(outputs["edge"].cpu(), batch["edge"])
                    curve_metrics = binary_metrics(outputs["curve"].cpu(), batch["curve"])
                    aggregate["edge_f1"] += edge_metrics["f1"]
                    aggregate["curve_f1"] += curve_metrics["f1"]
            epoch_row["val_edge_f1"] = aggregate["edge_f1"] / len(val_loader)
            epoch_row["val_curve_f1"] = aggregate["curve_f1"] / len(val_loader)
            validation_f1 = epoch_row["val_curve_f1"]

        append_jsonl(output_dir / "metrics.jsonl", epoch_row)
        payload = _checkpoint_payload(
            model, optimizer, scheduler, scaler, stage, epoch, global_step, config, max(best_f1, validation_f1)
        )
        atomic_torch_save(payload, output_dir / "last.pt")
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            payload["best_f1"] = best_f1
            atomic_torch_save(payload, output_dir / "best.pt")
            _save_lightweight_weights(model, output_dir, "best")
        print(epoch_row)

    return output_dir / "best.pt"
