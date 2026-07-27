from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    image: Array
    target: Array


def _canvas(size: int, value: float = 0.75) -> Array:
    return np.full((size, size, 3), value, dtype=np.float32)


def _draw_gt_line(gt: Array, points: Array, thickness: int = 1, closed: bool = False) -> None:
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(gt, [pts], closed, 255, thickness, cv2.LINE_AA)


def _draw_rgb_line(image: Array, points: Array, color: tuple[float, float, float], thickness: int, closed: bool = False) -> None:
    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], closed, color, thickness, cv2.LINE_AA)


def _bezier(p0: Array, p1: Array, p2: Array, p3: Array, n: int = 180) -> Array:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _degrade(image: Array, rng: np.random.Generator, kind: str) -> Array:
    x = image.copy()
    h, w = x.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    if kind in {"illumination", "mixed"}:
        illumination = 0.62 + 0.46 * (xx / max(w - 1, 1)) + 0.13 * np.sin(2 * np.pi * yy / h)
        x *= illumination[..., None]
    if kind in {"texture", "mixed"}:
        texture = 0.045 * np.sin(2 * np.pi * xx / 7.0) + 0.035 * np.sin(2 * np.pi * (xx + yy) / 13.0)
        x += texture[..., None]
    if kind in {"noise", "mixed"}:
        x += rng.normal(0.0, 0.035 if kind == "noise" else 0.025, x.shape).astype(np.float32)
    if kind in {"blur", "mixed"}:
        x = cv2.GaussianBlur(x, (0, 0), 1.15 if kind == "blur" else 0.75, borderType=cv2.BORDER_REFLECT101)
    x = np.clip(x, 0.0, 1.0)
    if kind == "jpeg":
        bgr = np.clip(x[..., ::-1] * 255.0, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 45])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        x = cv2.imdecode(encoded, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    return x


def make_suite(size: int = 192, seed: int = 1234) -> list[SyntheticCase]:
    rng = np.random.default_rng(seed)
    cases: list[SyntheticCase] = []
    degradations = ["clean", "illumination", "noise", "blur", "texture", "jpeg", "mixed"]
    for index, degradation in enumerate(degradations):
        image = _canvas(size, 0.74)
        gt = np.zeros((size, size), np.uint8)
        center = np.array([size * 0.28, size * 0.32], np.float32)
        for angle in np.linspace(0.0, np.pi, 8, endpoint=False):
            radius = size * 0.22
            end = center + radius * np.array([np.cos(angle), np.sin(angle)], np.float32)
            pts = np.stack([center, end])
            color = (0.08, 0.11, 0.13) if index % 2 == 0 else (0.12, 0.22, 0.08)
            _draw_rgb_line(image, pts, color, 1 + (index % 2))
            _draw_gt_line(gt, pts, 1)
        curve = _bezier(
            np.array([12, size * 0.72]),
            np.array([size * 0.32, size * 0.48]),
            np.array([size * 0.64, size * 0.96]),
            np.array([size - 12, size * 0.66]),
        )
        _draw_rgb_line(image, curve, (0.04, 0.05, 0.06), 2)
        _draw_gt_line(gt, curve, 1)
        curve2 = _bezier(
            np.array([size * 0.52, 12]),
            np.array([size * 0.68, size * 0.22]),
            np.array([size * 0.78, size * 0.05]),
            np.array([size - 10, size * 0.30]),
        )
        _draw_rgb_line(image, curve2, (0.97, 0.97, 0.95), 3)
        _draw_gt_line(gt, curve2, 1)
        pts = np.array([
            [size * 0.57, size * 0.40],
            [size * 0.88, size * 0.43],
            [size * 0.82, size * 0.58],
            [size * 0.60, size * 0.55],
        ])
        _draw_rgb_line(image, pts, (0.18, 0.08, 0.25), 2, closed=True)
        _draw_gt_line(gt, pts, 1, closed=True)
        degraded = image if degradation == "clean" else _degrade(image, rng, degradation)
        cases.append(SyntheticCase(degradation, np.clip(degraded, 0.0, 1.0), gt >= 64))

    image = _canvas(size, 0.50)
    gt = np.zeros((size, size), np.uint8)
    isoluminant = (1.0, float((0.50 - 0.2126) / 0.7152), 0.0)
    curve = _bezier(
        np.array([10, size * 0.18]),
        np.array([size * 0.28, size * 0.95]),
        np.array([size * 0.72, size * 0.05]),
        np.array([size - 10, size * 0.82]),
    )
    _draw_rgb_line(image, curve, isoluminant, 2)
    _draw_gt_line(gt, curve, 1)
    cases.append(SyntheticCase("isoluminant_color", image, gt >= 64))

    image = _canvas(size, 0.58)
    gt = np.zeros((size, size), np.uint8)
    faint_a = _bezier(
        np.array([8, size * 0.30]),
        np.array([size * 0.30, size * 0.12]),
        np.array([size * 0.64, size * 0.55]),
        np.array([size - 8, size * 0.28]),
    )
    faint_b = _bezier(
        np.array([10, size * 0.72]),
        np.array([size * 0.38, size * 0.48]),
        np.array([size * 0.72, size * 0.98]),
        np.array([size - 12, size * 0.68]),
    )
    _draw_rgb_line(image, faint_a, (0.49, 0.49, 0.49), 1)
    _draw_rgb_line(image, faint_b, (0.67, 0.67, 0.67), 2)
    _draw_gt_line(gt, faint_a, 1)
    _draw_gt_line(gt, faint_b, 1)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    image *= (0.90 + 0.18 * xx / max(size - 1, 1))[..., None]
    image += rng.normal(0.0, 0.008, image.shape).astype(np.float32)
    cases.append(SyntheticCase("faint_lines", np.clip(image, 0.0, 1.0), gt >= 64))

    return cases
