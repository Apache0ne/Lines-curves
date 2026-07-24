from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from lines_curves.trainer import train_stage
from lines_curves.utils import load_yaml


def run_all_stages(
    config: dict[str, Any], start_stage: int = 1, auto_resume: bool = False
) -> Path:
    if start_stage not in (1, 2, 3):
        raise ValueError("start_stage must be 1, 2, or 3")
    output_root = Path(config["common"]["output_root"]).expanduser().resolve()
    checkpoint: Path | None = None
    for stage in range(start_stage, 4):
        stage_dir = output_root / f"stage{stage}"
        last = stage_dir / "last.pt"
        best = stage_dir / "best.pt"
        resume: Path | None = None
        if auto_resume and last.exists():
            payload = torch.load(last, map_location="cpu", weights_only=False)
            if int(payload.get("stage", -1)) != stage:
                raise RuntimeError(f"Checkpoint stage mismatch: expected {stage}, got {payload.get('stage')}")
            completed_epochs = int(payload.get("epoch", -1)) + 1
            required_epochs = int(config[f"stage{stage}"]["epochs"])
            if completed_epochs >= required_epochs and best.exists():
                checkpoint = best
                print(
                    f"STAGE_{stage}_STATUS=ALREADY_COMPLETE "
                    f"EPOCHS={completed_epochs}/{required_epochs} BEST={best}"
                )
                continue
            resume = last
            print(
                f"STAGE_{stage}_STATUS=RESUMING "
                f"EPOCHS={completed_epochs}/{required_epochs} CHECKPOINT={last}"
            )
        checkpoint = train_stage(config, stage, resume=resume)
    if checkpoint is None:
        raise RuntimeError("No stages were selected")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stages 1, 2, and 3 in sequence")
    parser.add_argument("--config", default="configs/colab.yaml")
    parser.add_argument("--start-stage", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume each incomplete stage from last.pt and skip completed stages.",
    )
    args = parser.parse_args()
    checkpoint = run_all_stages(load_yaml(args.config), args.start_stage, args.auto_resume)
    print(f"FINAL_CHECKPOINT={Path(checkpoint).resolve()}")


if __name__ == "__main__":
    main()
