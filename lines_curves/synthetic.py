from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageFilter


class CurvePointBank:
    """Lazy index of CurveML-like CSV point sets.

    Any CSV containing at least two numeric columns is accepted. Header and
    nonnumeric columns are ignored. This makes the loader tolerant of CurveML
    family and Bézier sub-dataset layouts.
    """

    def __init__(self, root: str | Path | None) -> None:
        self.root = Path(root).expanduser().resolve() if root else None
        self.files: list[Path] = []
        if self.root and self.root.exists():
            manifest = self.root / "point_manifest.txt"
            if manifest.exists():
                self.files = [
                    self.root / line.strip()
                    for line in manifest.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                self.files = [path for path in self.files if path.exists()]
            else:
                self.files = sorted(self.root.rglob("*.csv"))

    def __len__(self) -> int:
        return len(self.files)

    def sample(self, rng: np.random.Generator) -> np.ndarray | None:
        if not self.files:
            return None
        for _ in range(4):
            path = self.files[int(rng.integers(0, len(self.files)))]
            points = self._read_points(path)
            if points is not None and len(points) >= 8:
                return points
        return None

    @staticmethod
    @lru_cache(maxsize=2048)
    def _read_points(path: Path) -> np.ndarray | None:
        rows: list[list[float]] = []
        try:
            with path.open("r", newline="", encoding="utf-8", errors="ignore") as handle:
                for row in csv.reader(handle):
                    numeric: list[float] = []
                    for item in row:
                        try:
                            numeric.append(float(item))
                        except ValueError:
                            continue
                    if len(numeric) >= 2:
                        rows.append(numeric[:2])
        except OSError:
            return None
        if len(rows) < 8:
            return None
        points = np.asarray(rows, dtype=np.float32)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        return points if len(points) >= 8 else None


def _bezier(control: np.ndarray, count: int = 256) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    if len(control) == 3:
        return (1 - t) ** 2 * control[0] + 2 * (1 - t) * t * control[1] + t**2 * control[2]
    return (
        (1 - t) ** 3 * control[0]
        + 3 * (1 - t) ** 2 * t * control[1]
        + 3 * (1 - t) * t**2 * control[2]
        + t**3 * control[3]
    )


def _procedural_curve(rng: np.random.Generator) -> np.ndarray:
    family = int(rng.integers(0, 8))
    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    if family <= 2:
        degree = 3 if family < 2 else 2
        control = rng.uniform(-1.0, 1.0, size=(degree + 1, 2)).astype(np.float32)
        control[0] = rng.uniform(-1.0, -0.4, 2)
        control[-1] = rng.uniform(0.4, 1.0, 2)
        return _bezier(control)
    if family == 3:
        angle = rng.uniform(np.pi * 0.4, np.pi * 1.8)
        theta = np.linspace(rng.uniform(0, 2 * np.pi), angle, 256, dtype=np.float32)
        return np.stack([np.cos(theta), np.sin(theta)], axis=1)
    if family == 4:
        theta = np.linspace(0, 2 * np.pi, 256, dtype=np.float32)
        ratio = rng.uniform(0.3, 0.9)
        return np.stack([np.cos(theta), ratio * np.sin(theta)], axis=1)
    if family == 5:
        theta = np.linspace(0, rng.uniform(2.5, 5.5) * np.pi, 256, dtype=np.float32)
        radius = np.linspace(0.1, 1.0, 256, dtype=np.float32)
        return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    if family == 6:
        x = np.linspace(-1, 1, 256, dtype=np.float32)
        y = np.sin(x * rng.uniform(1.5, 3.5) * np.pi) * rng.uniform(0.25, 0.8)
        return np.stack([x, y], axis=1)
    theta = np.linspace(0, 2 * np.pi, 256, dtype=np.float32)
    petals = int(rng.integers(2, 8))
    radius = np.cos(petals * theta)
    return np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)


def _normalize_and_place(
    points: np.ndarray, rng: np.random.Generator, size: int, margin: int = 12
) -> np.ndarray:
    points = points.astype(np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    span = np.ptp(points, axis=0)
    points = points / max(float(span.max()), 1e-6)
    angle = float(rng.uniform(-np.pi, np.pi))
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    points = points @ rotation.T
    scale = float(rng.uniform(0.25, 0.85)) * (size - 2 * margin)
    points *= scale
    center = rng.uniform(margin + scale * 0.45, size - margin - scale * 0.45, size=2)
    points += center
    return np.round(points).astype(np.int32)


def _background(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    base = rng.integers(20, 236, size=3, dtype=np.uint8)
    image = np.empty((size, size, 3), dtype=np.uint8)
    image[:] = base
    structural_edge = np.zeros((size, size), dtype=np.uint8)
    # Low-frequency texture.
    noise_small = rng.normal(0, rng.uniform(5, 35), size=(max(4, size // 16), max(4, size // 16), 3))
    noise = cv2.resize(noise_small.astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    for _ in range(int(rng.integers(0, 8))):
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        p1_raw = tuple(int(v) for v in rng.integers(0, size, size=2))
        p2_raw = tuple(int(v) for v in rng.integers(0, size, size=2))
        p1 = (min(p1_raw[0], p2_raw[0]), min(p1_raw[1], p2_raw[1]))
        p2 = (max(p1_raw[0], p2_raw[0]), max(p1_raw[1], p2_raw[1]))
        cv2.rectangle(image, p1, p2, color, thickness=-1)
        cv2.rectangle(structural_edge, p1, p2, 255, thickness=1, lineType=cv2.LINE_AA)
    image = cv2.GaussianBlur(image, (0, 0), sigmaX=float(rng.uniform(0.0, 2.0)))
    return image, structural_edge


def render_composite(
    rng: np.random.Generator,
    size: int,
    point_bank: CurvePointBank | None = None,
    min_curves: int = 1,
    max_curves: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image, edge = _background(rng, size)
    curve = np.zeros_like(edge)

    curve_count = int(rng.integers(min_curves, max_curves + 1))
    for _ in range(curve_count):
        points = point_bank.sample(rng) if point_bank is not None else None
        if points is None:
            points = _procedural_curve(rng)
        points = _normalize_and_place(points, rng, size)
        thickness = int(rng.integers(1, 5))
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        cv2.polylines(image, [points], False, color, thickness, lineType=cv2.LINE_AA)
        cv2.polylines(edge, [points], False, 255, thickness, lineType=cv2.LINE_AA)
        cv2.polylines(curve, [points], False, 255, thickness, lineType=cv2.LINE_AA)

    # Straight lines are hard negatives for the curve head but positives for edge.
    for _ in range(int(rng.integers(1, 8))):
        p1 = tuple(int(v) for v in rng.integers(0, size, size=2))
        p2 = tuple(int(v) for v in rng.integers(0, size, size=2))
        thickness = int(rng.integers(1, 5))
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        cv2.line(image, p1, p2, color, thickness, lineType=cv2.LINE_AA)
        cv2.line(edge, p1, p2, 255, thickness, lineType=cv2.LINE_AA)

    # Partial occlusion forces continuity reasoning without labeling hidden pixels.
    for _ in range(int(rng.integers(0, 4))):
        x1, y1 = (int(v) for v in rng.integers(0, size - 8, size=2))
        x2 = min(size, x1 + int(rng.integers(6, max(7, size // 4))))
        y2 = min(size, y1 + int(rng.integers(6, max(7, size // 4))))
        patch_color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        cv2.rectangle(image, (x1, y1), (x2, y2), patch_color, -1)
        edge[y1 : y2 + 1, x1 : x2 + 1] = 0
        curve[y1 : y2 + 1, x1 : x2 + 1] = 0
        cv2.rectangle(edge, (x1, y1), (x2, y2), 255, 1, lineType=cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if rng.random() < 0.45:
        pil = pil.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.0, 1.4))))
    image = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    if rng.random() < 0.6:
        noise = rng.normal(0, rng.uniform(1, 12), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, edge, curve
