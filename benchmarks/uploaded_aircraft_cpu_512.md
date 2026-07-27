# Uploaded aircraft CPU benchmark

This benchmark was run locally on CPU against 27 user-provided natural aircraft photographs. The archive contained captions, but no pixel-level line masks, so these numbers measure transformation consistency rather than ground-truth line accuracy. The source images and captions are not redistributed.

## Setup

- Long side: 512 pixels
- Workers: 4
- Boundary matching tolerance: 1.5 pixels
- OpenCV: 4.13.0
- NumPy: 2.3.5
- Total wall time: 23.956 seconds
- Mean base-image runtime: 0.293 seconds

## Results

| Transformation | Mean F1 | Minimum F1 | Probability correlation |
|---|---:|---:|---:|
| Exact repeat | 100.0000% | 100.0000% | 1.000000 |
| Horizontal flip | 100.0000% | 100.0000% | 1.000000 |
| Vertical flip | 100.0000% | 100.0000% | 1.000000 |
| 90-degree rotation | 100.0000% | 100.0000% | 1.000000 |
| Exposure/gamma change | 98.0886% | 94.1229% | 0.945766 |
| JPEG quality 70 | 95.7454% | 92.2111% | 0.817236 |
| Gaussian noise, sigma 2/255 | 98.9439% | 97.6383% | 0.979917 |
| 75% resize and restore | 97.8033% | 95.5524% | 0.945013 |

Across the 189 non-repeat image/transformation pairs, mean F1 was 98.6545%, the minimum was 92.2111%, and 81 pairs were exactly 100%.

## CPU defect found and fixed

The previous hysteresis implementation repeatedly dilated strong pixels through the weak mask until convergence. That is correct but can require a number of full-image passes proportional to the connected component diameter. Dense natural-image cases caused extreme stalls.

The replacement labels each 8-connected weak component once and keeps components containing at least one strong pixel. It is mathematically equivalent to the iterative operation.

Validation:

- 300 randomized equivalence cases: PASS
- 256x256 long-propagation output: identical
- Previous implementation: 140.785 ms
- Connected-component implementation: 0.633 ms
- Isolated hysteresis speedup: approximately 222x

## Accuracy limitation

A 100% transformation-consistency score under JPEG compression, resampling, exposure changes, or noise is not a valid universal target: those operations alter or destroy image evidence. Producing an invariant result can be forced by discarding fine lines, which would make the detector less accurate. Ground-truth precision, recall, ODS, OIS, and AP require human or dataset-provided line masks.
