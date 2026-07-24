from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_data import prepare_biped, prepare_bsds


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Unsafe path in archive {archive_path}: {member.filename}")
        archive.extractall(destination)
    (destination / ".extract_complete").write_text(str(archive_path), encoding="utf-8")


def resolve_biped_root(downloaded: Path, extraction_root: Path) -> Path:
    """Extract Kaggle's nested BIPEDv2.zip when the dataset contains one."""
    if downloaded.is_file() and downloaded.suffix.lower() == ".zip":
        archives = [downloaded]
    else:
        archives = sorted(downloaded.rglob("*.zip"))
    if not archives:
        return downloaded
    archives.sort(
        key=lambda path: (
            0 if path.stem.lower() == "bipedv2" else 1,
            0 if "bipedv2" in path.name.lower() else 1,
            str(path),
        )
    )
    archive = archives[0]
    destination = extraction_root / archive.stem
    marker = destination / ".extract_complete"
    if not marker.exists() or marker.read_text(encoding="utf-8", errors="ignore") != str(archive):
        _safe_extract_zip(archive, destination)
    return destination


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

        downloaded_biped = Path(kagglehub.dataset_download("xavysp/biped")).resolve()
        biped_root = resolve_biped_root(downloaded_biped, download_root / "biped_extracted")
        print(f"BIPED_SOURCE={biped_root}")
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
