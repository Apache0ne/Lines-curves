from __future__ import annotations

import cv2
import numpy as np

from .core import _as_float_image, _fft_filter, _nms, _robust_unit

Array = np.ndarray


def _gray(image: Array) -> Array:
    x = _as_float_image(image)
    if x.ndim == 2:
        return x
    return (0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]).astype(np.float32)


def sobel_probability(image: Array) -> Array:
    gray = cv2.GaussianBlur(_gray(image), (0, 0), 1.0, borderType=cv2.BORDER_REFLECT101)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0, borderType=cv2.BORDER_REFLECT101)
    response = _nms(np.hypot(gx, gy), gx, gy)
    return _robust_unit(response, 99.7)


def scharr_probability(image: Array) -> Array:
    gray = cv2.GaussianBlur(_gray(image), (0, 0), 1.0, borderType=cv2.BORDER_REFLECT101)
    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0, scale=1.0 / 32.0, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1, scale=1.0 / 32.0, borderType=cv2.BORDER_REFLECT101)
    response = _nms(np.hypot(gx, gy), gx, gy)
    return _robust_unit(response, 99.7)


def fourier_scharr_probability(image: Array) -> Array:
    gray = _gray(image)
    filtered = _fft_filter(gray, highpass=0.012, lowpass=0.42, order=4)
    filtered = cv2.GaussianBlur(filtered, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT101)
    gx = cv2.Scharr(filtered, cv2.CV_32F, 1, 0, scale=1.0 / 32.0, borderType=cv2.BORDER_REFLECT101)
    gy = cv2.Scharr(filtered, cv2.CV_32F, 0, 1, scale=1.0 / 32.0, borderType=cv2.BORDER_REFLECT101)
    response = _nms(np.hypot(gx, gy), gx, gy)
    return _robust_unit(response, 99.7)


def canny_binary(image: Array) -> Array:
    gray = np.clip(_gray(image) * 255.0, 0, 255).astype(np.uint8)
    median = float(np.median(gray))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    return cv2.Canny(gray, lower, upper, L2gradient=True) > 0
