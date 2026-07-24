from __future__ import annotations

import argparse
from pathlib import Path

from lines_curves.trainer import train_stage
from lines_curves.utils import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one TEED-Curves stage")
    parser.add_argument("--config", default="configs/colab.yaml")
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    checkpoint = train_stage(load_yaml(args.config), args.stage, args.resume)
    print(f"BEST_CHECKPOINT={Path(checkpoint).resolve()}")


if __name__ == "__main__":
    main()
