from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_data import prepare_biped, prepare_bsds


def clone_once(url: str, destination: Path, recurse_submodules: bool = False) -> None:
    if destination.exists() and any(destination.iterdir()):
        return
    command = ["git", "clone", "--depth=1"]
    if recurse_submodules:
        command.append("--recurse-submodules")
    command.extend([url, str(destination)])
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalize Colab training data")
    parser.add_argument("--download-root", default="/content/lines-curves-datasets")
    parser.add_argument("--natural-output", default=str(ROOT / "data" / "natural"))
    parser.add_argument("--curveml-output", default=str(ROOT / "data" / "curveml"))
    parser.add_argument("--skip-biped", action="store_true")
    parser.add_argument("--skip-bsds", action="store_true")
    parser.add_argument("--with-curveml", action="store_true")
    args = parser.parse_args()

    download_root = Path(args.download_root).expanduser().resolve()
    natural_output = Path(args.natural_output).expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    natural_output.mkdir(parents=True, exist_ok=True)

    if not args.skip_biped:
        import kagglehub

        biped_root = Path(kagglehub.dataset_download("xavysp/biped")).resolve()
        count = prepare_biped(biped_root, natural_output)
        print(f"BIPED_PREPARED={count}")

    if not args.skip_bsds:
        bsds_root = download_root / "BSDS500"
        clone_once("https://github.com/BIDS/BSDS500.git", bsds_root)
        count = prepare_bsds(bsds_root, natural_output, include_test=False)
        print(f"BSDS_PREPARED={count}")

    if args.with_curveml:
        curveml_output = Path(args.curveml_output).expanduser().resolve()
        clone_once("https://gitlab.com/4ndr3aR/CurveML.git", curveml_output, recurse_submodules=True)
        csv_files = sorted(curveml_output.rglob("*.csv"))
        manifest = curveml_output / "point_manifest.txt"
        manifest.write_text(
            "\n".join(str(path.relative_to(curveml_output)) for path in csv_files)
            + ("\n" if csv_files else ""),
            encoding="utf-8",
        )
        print(f"CURVEML_CSV_COUNT={len(csv_files)}")
        print(f"CURVEML_MANIFEST={manifest}")

    print(f"NATURAL_OUTPUT={natural_output}")


if __name__ == "__main__":
    main()
