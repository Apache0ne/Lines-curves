from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/teed")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    path = hf_hub_download(repo_id="fal/teed", filename="5_model.pth", local_dir=output)
    print(Path(path).resolve())


if __name__ == "__main__":
    main()
