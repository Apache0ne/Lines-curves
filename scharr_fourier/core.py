from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np
from scipy import ndimage

Array = np.ndarray
Mode = Literal["edge", "ridge", "hybrid"]


@dataclass(frozen=True)
class ScharrFourierConfig:
    mode: Mode = "hybrid"
    scales: tuple[float, ...] = (0.65, 1.0, 1.6, 2.5, 4.0)
    scale_weights: tuple[float, ...] = (1.0, 0.95, 0.85, 0.68, 0.48)
    fft_highpass: float = 0.012
    fft_lowpass: float = 0.42
    fft_order: int = 4
    illumination_sigma: float = 18.0
    local_contrast_sigma: float = 7.0
    color_weight: float = 0.38
    chroma_absolute_threshold: float = 0.02
    chroma_global_threshold: float = 0.06
    homomorphic_anchor_weight: float = 0.35
    adaptive_threshold: bool = True
    adaptive_low_ratio: float = 0.6
    adaptive_high_scale: float = 0.85
    adaptive_high_min: float = 0.12
    adaptive_high_max: float = 0.65
    coherence_weight: float = 0.32
    ridge_weight: float = 0.6
    edge_weight: float = 0.7
    nms: bool = True
    low_threshold: float = 0.3
    high_threshold: float = 0.52
    min_component: int = 12
    close_radius: int = 0
    return_thinned: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class LineResult:
    probability: Array
    binary: Array
    orientation: Array
    edge_response: Array
    ridge_response: Array
    coherence: Array
    anchor_response: Array
    fusion_response: Array


def _as_float_image(image: Array) -> Array:
    a = np.asarray(image)
    if a.ndim not in (2, 3):
        raise ValueError(f"Expected HxW or HxWxC image, got shape {a.shape}")
    if a.ndim == 3 and a.shape[2] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {a.shape}")
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
    else:
        a = a.astype(np.float32)
        finite = np.isfinite(a)
        if not finite.all():
            a = np.where(finite, a, 0.0)
        if a.size and (a.min() < 0.0 or a.max() > 1.0):
            lo, hi = np.percentile(a, [0.1, 99.9])
            a = np.clip((a - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    if a.ndim == 3 and a.shape[2] == 4:
        rgb, alpha = a[..., :3], a[..., 3:4]
        a = rgb * alpha + (1.0 - alpha)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    return np.ascontiguousarray(a, dtype=np.float32)


def _gray_and_opponent(image: Array) -> tuple[Array, tuple[Array, ...]]:
    a = _as_float_image(image)
    if a.ndim == 2:
        return a, ()
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    rg = (r - g) / np.sqrt(2.0)
    yb = (0.5 * (r + g) - b) / np.sqrt(1.5)
    return gray.astype(np.float32), (rg.astype(np.float32), yb.astype(np.float32))


def _reflect_pad(image: Array) -> tuple[Array, tuple[slice, slice]]:
    h, w = image.shape
    py = max(16, min(h // 2, int(round(h * 0.16))))
    px = max(16, min(w // 2, int(round(w * 0.16))))
    padded = np.pad(image, ((py, py), (px, px)), mode="reflect")
    return padded, (slice(py, py + h), slice(px, px + w))


def _radial_frequency(shape: tuple[int, int]) -> Array:
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    return np.sqrt(fx * fx + fy * fy).astype(np.float32)


def _fft_filter(image: Array, *, highpass: float, lowpass: float, order: int) -> Array:
    padded, crop = _reflect_pad(image)
    radius = _radial_frequency(padded.shape)
    eps = np.finfo(np.float32).eps
    if highpass > 0:
        safe_radius = np.maximum(radius, highpass * 1e-4)
        ratio = np.minimum(highpass / safe_radius, 1e4)
        hp = 1.0 / (1.0 + ratio ** (2 * order))
        hp[0, 0] = 0.0
    else:
        hp = np.ones_like(radius)
    if lowpass < 0.5:
        lp = 1.0 / (1.0 + (radius / max(lowpass, eps)) ** (2 * order))
    else:
        lp = np.ones_like(radius)
    spectrum = np.fft.rfft2(padded)
    filtered = np.fft.irfft2(spectrum * (hp * lp), s=padded.shape).real
    return filtered[crop].astype(np.float32)


def _homomorphic_normalize(gray: Array, cfg: ScharrFourierConfig) -> Array:
    x = np.log(np.clip(gray, 1e-4, 1.0))
    illumination = cv2.GaussianBlur(
        x, (0, 0), cfg.illumination_sigma, borderType=cv2.BORDER_REFLECT101
    )
    detail = x - illumination
    lo, hi = np.percentile(detail, [0.5, 99.5])
    detail = np.clip((detail - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return detail.astype(np.float32)


def _robust_unit(x: Array, q: float = 99.5) -> Array:
    x = np.maximum(x.astype(np.float32), 0.0)
    scale = float(np.percentile(x, q))
    if scale <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip(x / scale, 0.0, 1.0).astype(np.float32)


def _scharr(image: Array) -> tuple[Array, Array]:
    gx = cv2.Scharr(
        image, cv2.CV_32F, 1, 0, scale=1.0 / 32.0, borderType=cv2.BORDER_REFLECT101
    )
    gy = cv2.Scharr(
        image, cv2.CV_32F, 0, 1, scale=1.0 / 32.0, borderType=cv2.BORDER_REFLECT101
    )
    return gx, gy


def _nms(response: Array, gx: Array, gy: Array) -> Array:
    mag = response.astype(np.float32)
    norm = np.sqrt(gx * gx + gy * gy) + 1e-12
    dx = gx / norm
    dy = gy / norm
    yy, xx = np.mgrid[: mag.shape[0], : mag.shape[1]].astype(np.float32)
    p = cv2.remap(
        mag, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )
    n = cv2.remap(
        mag, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101
    )
    return np.where((mag >= p) & (mag >= n), mag, 0.0).astype(np.float32)


def _hessian_ridge(image: Array) -> tuple[Array, Array, Array]:
    gx, gy = _scharr(image)
    gxx, gxy_a = _scharr(gx)
    gxy_b, gyy = _scharr(gy)
    gxy = 0.5 * (gxy_a + gxy_b)
    trace = gxx + gyy
    delta = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4.0 * gxy * gxy, 0.0))
    l_small_alg = 0.5 * (trace - delta)
    l_large_alg = 0.5 * (trace + delta)
    small_abs_first = np.abs(l_small_alg) <= np.abs(l_large_alg)
    lambda_minor = np.where(small_abs_first, l_small_alg, l_large_alg)
    lambda_major = np.where(small_abs_first, l_large_alg, l_small_alg)
    rb = np.abs(lambda_minor) / (np.abs(lambda_major) + 1e-8)
    structureness = np.sqrt(lambda_minor * lambda_minor + lambda_major * lambda_major)
    nonzero = structureness[structureness > 0]
    c = float(np.percentile(nonzero, 90.0)) if nonzero.size else 1.0
    c = max(c, 1e-6)
    beta = 0.45
    ridge = np.exp(-(rb * rb) / (2.0 * beta * beta)) * (
        1.0 - np.exp(-(structureness * structureness) / (2.0 * c * c))
    )
    theta_large = 0.5 * np.arctan2(2.0 * gxy, gxx - gyy)
    theta_major = np.where(small_abs_first, theta_large, theta_large + 0.5 * np.pi)
    return (
        ridge.astype(np.float32),
        np.cos(theta_major).astype(np.float32),
        np.sin(theta_major).astype(np.float32),
    )


def _orientation_coherence(gx: Array, gy: Array, sigma: float = 1.4) -> Array:
    jxx = cv2.GaussianBlur(
        gx * gx, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101
    )
    jyy = cv2.GaussianBlur(
        gy * gy, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101
    )
    jxy = cv2.GaussianBlur(
        gx * gy, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101
    )
    delta = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4.0 * jxy * jxy, 0.0))
    return np.clip(delta / (jxx + jyy + 1e-8), 0.0, 1.0).astype(np.float32)


def _hysteresis(prob: Array, low: float, high: float) -> Array:
    """Keep 8-connected weak components containing at least one strong pixel.

    This is exactly equivalent to repeatedly dilating strong pixels through the
    weak mask, but it completes in one connected-component pass rather than a
    number of full-image iterations proportional to the component diameter.
    """
    if not (0.0 <= low <= high <= 1.0):
        raise ValueError("Thresholds must satisfy 0 <= low <= high <= 1")
    weak = prob >= low
    strong = prob >= high
    if not np.any(strong):
        return np.zeros_like(weak, dtype=bool)
    labels, count = ndimage.label(
        weak, structure=np.ones((3, 3), dtype=np.uint8)
    )
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return keep[labels]


def _thin(binary: Array) -> Array:
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        out = cv2.ximgproc.thinning(binary.astype(np.uint8) * 255)
        return out > 0
    img = binary.astype(np.uint8).copy()
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            p2 = np.roll(img, -1, axis=0)
            p3 = np.roll(np.roll(img, -1, axis=0), 1, axis=1)
            p4 = np.roll(img, 1, axis=1)
            p5 = np.roll(np.roll(img, 1, axis=0), 1, axis=1)
            p6 = np.roll(img, 1, axis=0)
            p7 = np.roll(np.roll(img, 1, axis=0), -1, axis=1)
            p8 = np.roll(img, -1, axis=1)
            p9 = np.roll(np.roll(img, -1, axis=0), -1, axis=1)
            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = sum(
                condition.astype(np.uint8)
                for condition in (
                    (p2 == 0) & (p3 == 1),
                    (p3 == 0) & (p4 == 1),
                    (p4 == 0) & (p5 == 1),
                    (p5 == 0) & (p6 == 1),
                    (p6 == 0) & (p7 == 1),
                    (p7 == 0) & (p8 == 1),
                    (p8 == 0) & (p9 == 1),
                    (p9 == 0) & (p2 == 1),
                )
            )
            if step == 0:
                c3 = (p2 * p4 * p6) == 0
                c4 = (p4 * p6 * p8) == 0
            else:
                c3 = (p2 * p4 * p8) == 0
                c4 = (p2 * p6 * p8) == 0
            remove = (
                (img == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & c3
                & c4
            )
            remove[[0, -1], :] = False
            remove[:, [0, -1]] = False
            if np.any(remove):
                img[remove] = 0
                changed = True
    return img.astype(bool)


def _direct_fourier_scharr(
    image: Array, cfg: ScharrFourierConfig
) -> tuple[Array, Array]:
    filtered = _fft_filter(
        image,
        highpass=cfg.fft_highpass,
        lowpass=cfg.fft_lowpass,
        order=cfg.fft_order,
    )
    filtered = cv2.GaussianBlur(
        filtered, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT101
    )
    gx, gy = _scharr(filtered)
    raw = _nms(np.hypot(gx, gy), gx, gy) if cfg.nms else np.hypot(gx, gy)
    return raw.astype(np.float32), _robust_unit(raw, 99.7)


def _adaptive_hysteresis(probability: Array, cfg: ScharrFourierConfig) -> Array:
    scaled = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    if not np.any(scaled):
        return np.zeros_like(probability, dtype=bool)
    threshold, _ = cv2.threshold(
        scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    high = float(
        np.clip(
            cfg.adaptive_high_scale * threshold / 255.0,
            cfg.adaptive_high_min,
            cfg.adaptive_high_max,
        )
    )
    low = float(np.clip(cfg.adaptive_low_ratio * high, 0.0, high))
    return _hysteresis(probability, low, high)


def detect_lines(
    image: Array, config: ScharrFourierConfig | None = None
) -> LineResult:
    cfg = config or ScharrFourierConfig()
    if len(cfg.scales) != len(cfg.scale_weights):
        raise ValueError("scales and scale_weights must have equal length")
    gray, opponents = _gray_and_opponent(image)
    base = _homomorphic_normalize(gray, cfg)
    base = _fft_filter(
        base,
        highpass=cfg.fft_highpass,
        lowpass=cfg.fft_lowpass,
        order=cfg.fft_order,
    )

    edge_acc = np.zeros_like(gray, dtype=np.float32)
    ridge_acc = np.zeros_like(gray, dtype=np.float32)
    gx_acc = np.zeros_like(gray, dtype=np.float32)
    gy_acc = np.zeros_like(gray, dtype=np.float32)
    coherence_acc = np.zeros_like(gray, dtype=np.float32)
    weight_sum = 0.0

    for sigma, weight in zip(cfg.scales, cfg.scale_weights):
        smooth = cv2.GaussianBlur(
            base, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101
        )
        gx, gy = _scharr(smooth)
        mag = np.hypot(gx, gy)
        coherence = _orientation_coherence(gx, gy, sigma=max(0.8, sigma))
        edge = mag * (1.0 + cfg.coherence_weight * coherence)
        if cfg.nms:
            edge = _nms(edge, gx, gy)
        ridge, nx, ny = _hessian_ridge(smooth)
        if cfg.nms:
            ridge = _nms(ridge, nx, ny)
        edge_acc = np.maximum(edge_acc, float(weight) * _robust_unit(edge))
        ridge_acc = np.maximum(ridge_acc, float(weight) * _robust_unit(ridge))
        gx_acc += float(weight) * gx
        gy_acc += float(weight) * gy
        coherence_acc += float(weight) * coherence
        weight_sum += float(weight)

    if opponents and cfg.color_weight > 0:
        chroma = np.zeros_like(gray, dtype=np.float32)
        for channel in opponents:
            filtered = _fft_filter(
                channel,
                highpass=cfg.fft_highpass,
                lowpass=cfg.fft_lowpass,
                order=cfg.fft_order,
            )
            gx, gy = _scharr(
                cv2.GaussianBlur(
                    filtered, (0, 0), 1.0, borderType=cv2.BORDER_REFLECT101
                )
            )
            response = np.hypot(gx, gy)
            if cfg.nms:
                response = _nms(response, gx, gy)
            chroma = np.maximum(chroma, _robust_unit(response))
        edge_acc = np.maximum(edge_acc, cfg.color_weight * chroma)

    coherence_acc /= max(weight_sum, 1e-8)
    edge_prob = _robust_unit(edge_acc, 99.7)
    ridge_prob = _robust_unit(ridge_acc, 99.7)
    e = np.clip(cfg.edge_weight * edge_prob, 0.0, 1.0)
    r = np.clip(cfg.ridge_weight * ridge_prob, 0.0, 1.0)
    fusion = 1.0 - (1.0 - e) * (1.0 - r)
    fusion *= 0.78 + 0.22 * np.sqrt(np.clip(coherence_acc, 0.0, 1.0))
    fusion = _robust_unit(fusion, 99.7)
    local_mean = cv2.GaussianBlur(
        fusion, (0, 0), cfg.local_contrast_sigma, borderType=cv2.BORDER_REFLECT101
    )
    local_sq = cv2.GaussianBlur(
        fusion * fusion,
        (0, 0),
        cfg.local_contrast_sigma,
        borderType=cv2.BORDER_REFLECT101,
    )
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))
    fusion = _robust_unit(
        np.maximum(fusion - 0.45 * local_mean, 0.0) / (0.18 + 0.65 * local_std),
        99.7,
    )

    _, gray_anchor = _direct_fourier_scharr(gray, cfg)
    _, homomorphic_anchor = _direct_fourier_scharr(
        _homomorphic_normalize(gray, cfg), cfg
    )
    anchor = np.maximum(
        gray_anchor, cfg.homomorphic_anchor_weight * homomorphic_anchor
    )

    chroma_anchor = np.zeros_like(gray, dtype=np.float32)
    chroma_binary = np.zeros_like(gray, dtype=bool)
    if opponents and cfg.color_weight > 0.0:
        for channel in opponents:
            raw_chroma, normalized_chroma = _direct_fourier_scharr(channel, cfg)
            nonzero = raw_chroma[raw_chroma > 0.0]
            global_strength = (
                float(np.percentile(nonzero, 99.7)) if nonzero.size else 0.0
            )
            if global_strength >= cfg.chroma_global_threshold:
                gated = np.where(
                    raw_chroma >= cfg.chroma_absolute_threshold,
                    normalized_chroma,
                    0.0,
                )
                chroma_anchor = np.maximum(chroma_anchor, gated)
                if cfg.adaptive_threshold:
                    chroma_binary |= _adaptive_hysteresis(
                        normalized_chroma, cfg
                    ) & (raw_chroma >= cfg.chroma_absolute_threshold)
        anchor = np.maximum(anchor, chroma_anchor)
    anchor = _robust_unit(anchor, 99.7)

    if cfg.mode == "edge":
        probability = edge_prob
        binary = _hysteresis(probability, cfg.low_threshold, cfg.high_threshold)
    elif cfg.mode == "ridge":
        probability = ridge_prob
        binary = _hysteresis(probability, cfg.low_threshold, cfg.high_threshold)
    elif cfg.mode == "hybrid":
        probability = anchor
        if cfg.adaptive_threshold:
            gray_binary = _adaptive_hysteresis(gray_anchor, cfg)
            homomorphic_binary = _adaptive_hysteresis(homomorphic_anchor, cfg)
            binary = gray_binary | homomorphic_binary | chroma_binary
        else:
            binary = _hysteresis(probability, cfg.low_threshold, cfg.high_threshold)
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")
    if cfg.close_radius > 0:
        k = 2 * cfg.close_radius + 1
        binary = (
            cv2.morphologyEx(
                binary.astype(np.uint8),
                cv2.MORPH_CLOSE,
                np.ones((k, k), np.uint8),
            )
            > 0
        )
    if cfg.min_component > 1:
        labels, _ = ndimage.label(
            binary, structure=np.ones((3, 3), dtype=np.uint8)
        )
        sizes = np.bincount(labels.ravel())
        keep = sizes >= cfg.min_component
        keep[0] = False
        binary = keep[labels]
    if cfg.return_thinned:
        binary = _thin(binary)

    orientation = np.mod(np.arctan2(gy_acc, gx_acc), np.pi).astype(np.float32)
    return LineResult(
        probability=probability.astype(np.float32),
        binary=binary.astype(bool),
        orientation=orientation,
        edge_response=edge_prob,
        ridge_response=ridge_prob,
        coherence=np.clip(coherence_acc, 0.0, 1.0).astype(np.float32),
        anchor_response=anchor.astype(np.float32),
        fusion_response=fusion.astype(np.float32),
    )
