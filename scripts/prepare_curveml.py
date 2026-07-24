from __future__ import annotations

import argparse
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lines_curves.synthetic import discover_point_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone or register the official CurveML dataset")
    parser.add_argument("--output", default="data/curveml")
    parser.add_argument("--source", default=None, help="Existing CurveML directory to copy/link")
    parser.add_argument("--clone", action="store_true", help="Clone the official GitLab repository")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    already_present = output.exists() and any(output.iterdir())
    if already_present:
        print(f"CURVEML_ALREADY_PRESENT={output}")
    elif args.source:
        source = Path(args.source).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copytree(source, output, dirs_exist_ok=True)
    elif args.clone:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "--recurse-submodules",
                "--shallow-submodules",
                "https://gitlab.com/4ndr3aR/CurveML.git",
                str(output),
            ],
            check=True,
        )
    elif not already_present:
        raise SystemExit("Use --source PATH or --clone")
    csv_files = discover_point_files(output)
    manifest = output / "point_manifest.txt"
    manifest.write_text(
        "\n".join(str(path.relative_to(output)) for path in csv_files) + ("\n" if csv_files else ""),
        encoding="utf-8",
    )
    csv_count = len(csv_files)
    print(f"CURVEML_ROOT={output}")
    print(f"CURVEML_MANIFEST={manifest}")
    print(f"CURVEML_POINT_SET_COUNT={csv_count}")
    if csv_count == 0:
        print("WARNING: no CSV point sets found; procedural curve generation will remain active.")


if __name__ == "__main__":
    main()
