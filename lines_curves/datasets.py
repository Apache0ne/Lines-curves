from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import CurvePointBank, render_composite


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class NaturalRecord:
    image: Path
    edge: Path
    curve: Path


def discover_natural_records(root: str | Path | None, split: str) -> list[NaturalRecord]:
    if root is None:
        return []
    root = Path(root).expanduser().resolve() / split
    image_dir, edge_dir, curve_dir = root / "images", root / "edges", root / "curves"
    if not all(path.exists() for path in (image_dir, edge_dir, curve_dir)):
        return []
    edge_by_stem = {path.stem: path for path in edge_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    curve_by_stem = {path.stem: path for path in curve_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    records = []
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if image.stem in edge_by_stem and image.stem in curve_by_stem:
            records.append(NaturalRecord(image, edge_by_stem[image.stem], curve_by_stem[image.stem]))
    return records


def _read_sample(record: NaturalRecord, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(str(record.image), cv2.IMREAD_COLOR)
    edge = cv2.imread(str(record.edge), cv2.IMREAD_GRAYSCALE)
    curve = cv2.imread(str(record.curve), cv2.IMREAD_GRAYSCALE)
    if image is None or edge is None or curve is None:
        raise OSError(f"Failed to load natural record: {record}")
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    edge = cv2.resize(edge, (size, size), interpolation=cv2.INTER_NEAREST)
    curve = cv2.resize(curve, (size, size), interpolation=cv2.INTER_NEAREST)
    return image, edge, curve


def _augment(
    image: np.ndarray, edge: np.ndarray, curve: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image, edge, curve = image[:, ::-1], edge[:, ::-1], curve[:, ::-1]
    if rng.random() < 0.25:
        image, edge, curve = image[::-1], edge[::-1], curve[::-1]
    k = int(rng.integers(0, 4))
    if k:
        image, edge, curve = np.rot90(image, k), np.rot90(edge, k), np.rot90(curve, k)
    if rng.random() < 0.5:
        alpha = float(rng.uniform(0.8, 1.2))
        beta = float(rng.uniform(-20, 20))
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image), np.ascontiguousarray(edge), np.ascontiguousarray(curve)


def _to_tensor(image: np.ndarray, edge: np.ndarray, curve: np.ndarray) -> dict[str, torch.Tensor]:
    # Exact original TEED preprocessing: BGR pixels minus the BIPED mean.
    mean = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)
    image_f = image.astype(np.float32) - mean
    image_t = torch.from_numpy(image_f.transpose(2, 0, 1).copy()).float()
    edge_t = torch.from_numpy((edge.astype(np.float32) / 255.0)[None]).float()
    curve_t = torch.from_numpy((curve.astype(np.float32) / 255.0)[None]).float()
    edge_t = (edge_t > 0.1).float()
    curve_t = (curve_t > 0.1).float() * edge_t
    return {"image": image_t, "edge": edge_t, "curve": curve_t}


class MixedCurveDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        size: int,
        samples_per_epoch: int,
        synthetic_fraction: float,
        seed: int,
        natural_root: str | Path | None = None,
        split: str = "train",
        curveml_root: str | Path | None = None,
        augment: bool = True,
    ) -> None:
        if not 0.0 <= synthetic_fraction <= 1.0:
            raise ValueError("synthetic_fraction must be between 0 and 1")
        self.size = size
        self.samples_per_epoch = samples_per_epoch
        self.synthetic_fraction = synthetic_fraction
        self.seed = seed
        self.epoch = 0
        self.augment = augment
        self.records = discover_natural_records(natural_root, split)
        self.point_bank = CurvePointBank(curveml_root)
        if synthetic_fraction < 1.0 and not self.records:
            raise FileNotFoundError(
                f"No normalized natural records found for split={split!r}. "
                "Run scripts/prepare_data.py first or set synthetic_fraction=1.0."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        use_synthetic = rng.random() < self.synthetic_fraction or not self.records
        if use_synthetic:
            image, edge, curve = render_composite(rng, self.size, self.point_bank)
        else:
            record = self.records[int(rng.integers(0, len(self.records)))]
            image, edge, curve = _read_sample(record, self.size)
        if self.augment:
            image, edge, curve = _augment(image, edge, curve, rng)
        return _to_tensor(image, edge, curve)


class NaturalValidationDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, root: str | Path, split: str, size: int) -> None:
        self.records = discover_natural_records(root, split)
        self.size = size
        if not self.records:
            raise FileNotFoundError(f"No natural validation records found under {Path(root) / split}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return _to_tensor(*_read_sample(self.records[index], self.size))
