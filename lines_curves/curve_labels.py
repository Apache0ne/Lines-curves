from __future__ import annotations

import cv2
import numpy as np


def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    """OpenCV-only morphological skeletonization for offline pseudo-labels."""
    image = (binary > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    max_iterations = image.shape[0] + image.shape[1]
    for _ in range(max_iterations):
        eroded = cv2.erode(image, element)
        opened = cv2.dilate(eroded, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = eroded
        if cv2.countNonZero(image) == 0:
            break
    return skeleton


def derive_curve_mask(
    edge_mask: np.ndarray,
    window: int = 8,
    angle_threshold_degrees: float = 12.0,
    min_contour_length: int = 24,
    dilation: int = 1,
) -> np.ndarray:
    """Derive curved portions from an edge mask using contour tangent change.

    This is deliberately conservative: short contours and nearly straight runs
    remain negative. It is intended to make natural-image pseudo-labels; exact
    synthetic curve masks are generated directly by ``synthetic.py``.
    """
    binary = (edge_mask > 0).astype(np.uint8) * 255
    centerline = skeletonize_binary(binary)
    contours, _ = cv2.findContours(centerline, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    result = np.zeros_like(binary)
    threshold = np.deg2rad(angle_threshold_degrees)
    for contour in contours:
        points = contour[:, 0, :].astype(np.float32)
        n = len(points)
        if n < max(min_contour_length, 2 * window + 1):
            continue
        closed = np.linalg.norm(points[0] - points[-1]) <= 2.0
        indices = range(n) if closed else range(window, n - window)
        selected: list[tuple[int, int]] = []
        for i in indices:
            i0 = (i - window) % n if closed else i - window
            i2 = (i + window) % n if closed else i + window
            v1 = points[i] - points[i0]
            v2 = points[i2] - points[i]
            norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if norm < 1e-6:
                continue
            cosine = float(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))
            angle = float(np.arccos(cosine))
            arc = float(np.linalg.norm(points[i2] - points[i0]))
            if angle >= threshold and arc >= window:
                selected.append((int(points[i, 0]), int(points[i, 1])))
        for point in selected:
            cv2.circle(result, point, max(1, dilation), 255, -1, lineType=cv2.LINE_AA)
    result = cv2.bitwise_and(result, cv2.dilate(binary, np.ones((3, 3), np.uint8)))
    return result
