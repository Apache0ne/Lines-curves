# Scharr + Fourier Lines

A deterministic, training-free line and curve extractor built around Scharr derivatives and reflect-padded Fourier filtering.

## Pipeline

The default `hybrid` path combines:

- Butterworth Fourier band-pass filtering with reflection padding to avoid wraparound seams;
- direct Scharr anchors on the original luminance signal;
- homomorphic illumination normalization and a second Fourier-Scharr anchor;
- gated opponent-color Scharr responses for isoluminant chromatic boundaries;
- adaptive Otsu-seeded hysteresis;
- multiscale Scharr gradients and structure-tensor coherence;
- a Scharr-derived Hessian/Frangi ridge branch for line centers;
- interpolated non-maximum suppression, component filtering, and optional Zhang-Suen thinning.

The returned object contains:

- `probability` and `binary` final line maps;
- `orientation`;
- `anchor_response` and `fusion_response`;
- separate `edge_response`, `ridge_response`, and `coherence` maps.

## Install

```bash
python -m pip install -e .
```

## Extract lines

```bash
python scripts/extract_lines.py input.png outputs/example
```

Python API:

```python
import cv2
from scharr_fourier import detect_lines

bgr = cv2.imread("input.png", cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
result = detect_lines(rgb)
cv2.imwrite("line_probability.png", (result.probability * 255).astype("uint8"))
cv2.imwrite("line_binary.png", result.binary.astype("uint8") * 255)
```

## Reproducible synthetic benchmark

```bash
python scripts/benchmark.py --size 192 --tolerance 2.5 --output outputs/benchmark
pytest
```

Recorded local result for the deterministic nine-case suite (`seed=1234`, 192×192, 2.5-pixel tolerance):

| Metric | Result |
|---|---:|
| Mean precision | 98.8% |
| Mean recall | 98.6% |
| Mean F1 | 98.8% |
| Minimum case F1 | 97.3% |
| Isoluminant-color case | 100% F1 |
| Faint-line case | 100% F1 |

The cases cover clean line art, uneven illumination, Gaussian noise, blur, periodic texture, JPEG damage, mixed corruption, isoluminant color boundaries, and faint lines. The exact machine-readable report is committed under `benchmarks/`.

This is **not** a claim of universal 100% accuracy. The current suite does not reach 100% on every corruption, and synthetic results are not interchangeable with official BSDS500 or BIPED scores.

## Evaluate paired datasets

For datasets exported as matching image and binary-target files:

```bash
python scripts/benchmark_pairs.py \
  --images /path/to/images \
  --targets /path/to/targets \
  --output outputs/dataset_report.json
```

Use each public dataset's official evaluator for publication-quality BSDS500/BIPED ODS, OIS, and AP numbers. The paired evaluator is a deterministic tolerance-aware diagnostic, not a replacement for official benchmark code.

## Tests and CI

GitHub Actions runs Python 3.10, 3.11, and 3.12, compiles the package, runs all tests, executes the synthetic benchmark, and uploads every generated comparison image and JSON report.
