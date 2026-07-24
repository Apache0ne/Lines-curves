from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import shutil
from typing import Any

import cv2
import numpy as np
from scipy.io import loadmat

from lines_curves.curve_labels import derive_curve_mask


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EDGE_WORDS = {"edge_maps", "gt", "groundtruth", "ground_truth", "label", "labels"}
IMAGE_WORDS = {"img", "imgs", "image", "images", "rgb"}
SUFFIXES = ("_edge", "-edge", "_gt", "-gt", "_label", "-label")


def normalized_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def infer_split(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if parts & {"test", "testing"}:
        return "test"
    if parts & {"val", "valid", "validation"}:
        return "val"
    return "train"


def copy_normalized(image: Path, edge: Path, output_root: Path, split: str, name: str) -> None:
    image_data = cv2.imread(str(image), cv2.IMREAD_COLOR)
    edge_data = cv2.imread(str(edge), cv2.IMREAD_GRAYSCALE)
    if image_data is None or edge_data is None:
        raise OSError(f"Failed to read pair: {image}, {edge}")
    if edge_data.shape != image_data.shape[:2]:
        edge_data = cv2.resize(edge_data, (image_data.shape[1], image_data.shape[0]), interpolation=cv2.INTER_NEAREST)
    curve = derive_curve_mask(edge_data)
    for folder in ("images", "edges", "curves"):
        (output_root / split / folder).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_root / split / "images" / f"{name}.png"), image_data)
    cv2.imwrite(str(output_root / split / "edges" / f"{name}.png"), edge_data)
    cv2.imwrite(str(output_root / split / "curves" / f"{name}.png"), curve)


def _replace_modality_path(image: Path, root: Path) -> list[Path]:
    """Return likely BIPED ground-truth paths for one RGB image."""
    try:
        relative = image.relative_to(root)
    except ValueError:
        relative = image
    parts = list(relative.parts)
    candidates: list[Path] = []
    for marker in ("imgs", "images"):
        if marker not in [part.lower() for part in parts]:
            continue
        marker_index = next(i for i, part in enumerate(parts) if part.lower() == marker)
        replaced = parts.copy()
        replaced[marker_index] = "edge_maps"
        base = root.joinpath(*replaced)
        for extension in (".png", ".jpg", ".jpeg", ".bmp"):
            candidates.append(base.with_suffix(extension))
    return candidates


def _biped_output_name(image: Path, root: Path) -> str:
    """Create a stable unique filename while preserving image/GT pairing."""
    try:
        relative = image.relative_to(root)
    except ValueError:
        relative = image
    parts = list(relative.with_suffix("").parts)
    lowered = [part.lower() for part in parts]
    for marker in ("imgs", "images"):
        if marker in lowered:
            parts = parts[lowered.index(marker) + 1 :]
            break
    safe = "_".join(parts)
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in safe)
    return f"biped_{safe.lower()}"


def prepare_biped(root: Path, output_root: Path) -> int:
    files = [path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    image_candidates = [
        path
        for path in files
        if ({part.lower() for part in path.parts} & IMAGE_WORDS)
        and "edge_maps" not in {part.lower() for part in path.parts}
    ]
    edge_candidates = [
        path
        for path in files
        if ({part.lower() for part in path.parts} & EDGE_WORDS)
        or any(word in path.stem.lower() for word in ("_edge", "-edge", "_gt", "-gt"))
    ]
    edge_index: dict[tuple[str, str], list[Path]] = {}
    for edge in edge_candidates:
        edge_index.setdefault((infer_split(edge), normalized_stem(edge)), []).append(edge)

    count = 0
    seen_outputs: set[tuple[str, str]] = set()
    for image in sorted(image_candidates):
        split, stem = infer_split(image), normalized_stem(image)
        counterpart = next((candidate for candidate in _replace_modality_path(image, root) if candidate.exists()), None)
        if counterpart is None:
            matches = edge_index.get((split, stem)) or edge_index.get(("train", stem))
            counterpart = matches[0] if matches else None
        if counterpart is None:
            continue
        name = _biped_output_name(image, root)
        identity = (split, name)
        if identity in seen_outputs:
            continue
        copy_normalized(image, counterpart, output_root, split, name)
        seen_outputs.add(identity)
        count += 1
    if count == 0:
        raise RuntimeError(
            f"No BIPED image/edge pairs were detected under {root}. "
            "Expected paths resembling edges/imgs/<split>/... and edges/edge_maps/<split>/...."
        )
    return count


def _find_boundary_arrays(value: Any, expected_hw: tuple[int, int]) -> list[np.ndarray]:
    found: list[np.ndarray] = []
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape == expected_hw and np.issubdtype(value.dtype, np.number):
            found.append(value.astype(np.float32))
        elif value.dtype == object:
            for item in value.flat:
                found.extend(_find_boundary_arrays(item, expected_hw))
    elif hasattr(value, "Boundaries"):
        found.extend(_find_boundary_arrays(value.Boundaries, expected_hw))
    elif hasattr(value, "__dict__"):
        for item in vars(value).values():
            found.extend(_find_boundary_arrays(item, expected_hw))
    return found


def prepare_bsds(root: Path, output_root: Path, include_test: bool = False) -> int:
    image_base_candidates = [root / "data" / "images", root / "BSDS500" / "data" / "images"]
    gt_base_candidates = [root / "data" / "groundTruth", root / "BSDS500" / "data" / "groundTruth"]
    image_base = next((path for path in image_base_candidates if path.exists()), None)
    gt_base = next((path for path in gt_base_candidates if path.exists()), None)
    if image_base is None or gt_base is None:
        raise FileNotFoundError("Expected BSDS500 data/images and data/groundTruth directories")
    count = 0
    for source_split, target_split in (("train", "train"), ("val", "val"), ("test", "test")):
        if source_split == "test" and not include_test:
            continue
        for image_path in sorted((image_base / source_split).glob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mat_path = gt_base / source_split / f"{image_path.stem}.mat"
            if not mat_path.exists():
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            arrays = _find_boundary_arrays(mat.get("groundTruth"), image.shape[:2])
            if not arrays:
                raise RuntimeError(f"Could not extract Boundaries arrays from {mat_path}")
            consensus = np.mean([(array > 0).astype(np.float32) for array in arrays], axis=0)
            edge = (consensus >= 0.20).astype(np.uint8) * 255
            temp_edge = output_root / ".tmp_bsds_edge.png"
            cv2.imwrite(str(temp_edge), edge)
            copy_normalized(image_path, temp_edge, output_root, target_split, f"bsds_{image_path.stem}")
            temp_edge.unlink(missing_ok=True)
            count += 1
    if count == 0:
        raise RuntimeError(f"No BSDS500 samples were prepared from {root}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize BIPED and BSDS500 for TEED-Curves")
    parser.add_argument("--output", default="data/natural")
    parser.add_argument("--biped-root", default=None)
    parser.add_argument("--bsds-root", default=None)
    parser.add_argument("--include-bsds-test", action="store_true")
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if args.clear and output.exists():
        shutil.rmtree(output)
    total = 0
    if args.biped_root:
        count = prepare_biped(Path(args.biped_root).expanduser().resolve(), output)
        print(f"BIPED_PREPARED={count}")
        total += count
    if args.bsds_root:
        count = prepare_bsds(Path(args.bsds_root).expanduser().resolve(), output, args.include_bsds_test)
        print(f"BSDS_PREPARED={count}")
        total += count
    if total == 0:
        raise SystemExit("Provide --biped-root and/or --bsds-root")
    print(f"TOTAL_PREPARED={total}")
    print(f"OUTPUT_ROOT={output}")


if __name__ == "__main__":
    main()
