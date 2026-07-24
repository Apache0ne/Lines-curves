from __future__ import annotations

import argparse
from pathlib import Path

from lines_curves.trainer import train_stage
from lines_curves.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stages 1, 2, and 3 in sequence")
    parser.add_argument("--config", default="configs/colab.yaml")
    parser.add_argument("--start-stage", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    config = load_yaml(args.config)
    checkpoint = None
    for stage in range(args.start_stage, 4):
        checkpoint = train_stage(config, stage)
    print(f"FINAL_CHECKPOINT={Path(checkpoint).resolve()}")


if __name__ == "__main__":
    main()
