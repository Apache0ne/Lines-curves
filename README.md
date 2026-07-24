# Lines-curves: TEED-Curves

A Colab-ready, tiny dual-output detector based on TEED. One shared encoder predicts:

1. a general perceptual edge map;
2. a curve-only map that rejects straight-line negatives.

The edge branch keeps the original TEED parameter names so the public `5_model.pth` checkpoint can initialize the shared encoder and edge decoder. The curve decoder is initialized from the trained edge decoder, then specialized.

## Training curriculum

| Stage | Trainable parameters | Data | Default schedule |
|---|---|---|---|
| 1 | Curve context and curve decoder only | 10,000 generated composites per epoch | 2 epochs |
| 2 | Entire model | CurveML/procedural curves + BIPED + BSDS500 | 6 epochs |
| 3 | Entire model, low LR | Natural BIPED + BSDS500 records only | 2 epochs |

Stage 1 automatically downloads the approximately 250 KB public TEED checkpoint from `fal/teed` on Hugging Face unless `common.teed_checkpoint` is set. It refuses to continue if too few compatible tensors load.

## Model size

The default model has **67,980 parameters**. The TEED-compatible shared encoder and edge branch contain **58,910 parameters**; the curve decoder and curve-context block add **9,070 parameters**.

A default-architecture export is approximately:

- `best.safetensors`: 277,808 bytes (FP32)
- `best_fp16.safetensors`: 141,840 bytes (FP16)

The resumable `.pt` files are larger because they also contain optimizer, scheduler, scaler, configuration, and Python/NumPy/Torch/CUDA RNG state.

## Colab notebook

Open [`Lines_Curves_Training_Colab.ipynb`](Lines_Curves_Training_Colab.ipynb) or follow [`COLAB.md`](COLAB.md). The notebook performs clone/update, installation, optional Google Drive configuration, dataset setup, preflight, all three training stages, and export.

## Fast verification

```bash
python -m pip install -r requirements.txt
python -m compileall -q .
pytest -q
python scripts/smoke_test.py
```

The test suite includes architecture compatibility, odd image sizes, finite gradients, BIPED/BSDS path normalization, CurveML compressed point sets, pseudo-label selectivity, full miniature Stage 1→2→3 handoff, exact interrupted-training resume, and automatic curriculum recovery.

## Data layout

The training loader expects normalized natural records:

```text
data/natural/
  train/
    images/*.png
    edges/*.png
    curves/*.png
  val/
    images/*.png
    edges/*.png
    curves/*.png
```

Create this layout from downloaded BIPED and BSDS500 directories:

```bash
python scripts/prepare_data.py \
  --biped-root /path/to/BIPED \
  --bsds-root /path/to/BSDS500 \
  --output data/natural \
  --clear
```

Natural curve pseudo-labels are generated conservatively from annotated edges by measuring tangent change along traced contours. Straight contours are excluded; curved contours remain. Synthetic curve targets are exact.

CurveML support accepts point-set files stored as `.csv` or `.csv.xz`. The manifest builder prioritizes `point_set_clean`/`point_cloud_clean` files so metadata CSVs and perturbed outlier sequences are not accidentally rendered as connected curves:

```bash
python scripts/prepare_curveml.py --clone
# or
python scripts/prepare_curveml.py --source /path/to/CurveML
```

If no usable CurveML point-set files are present, the built-in generator produces Bézier curves, arcs, ellipses, spirals, waves, and petal curves, with straight lines retained as hard negatives.

## Automatic Colab data setup

```bash
python scripts/colab_setup.py --with-curveml
```

This downloads public BIPED data through KaggleHub, safely extracts the nested BIPED archive when present, clones the BIDS BSDS500 mirror, derives natural curve pseudo-labels, and optionally clones CurveML. Dataset files are never committed to this repository.

## Preflight before training

```bash
python scripts/preflight.py \
  --config configs/colab.yaml \
  --download-teed \
  --report outputs/preflight.json
```

Preflight validates all paths and stage values, counts natural and CurveML records, verifies the TEED checkpoint, checks the 67,980-parameter architecture, validates the threshold grid, and runs one finite forward/backward sample for every stage. It exits nonzero on a blocking problem.

## Train

```bash
python train_all.py --config configs/colab.yaml --auto-resume
```

`--auto-resume` resumes an incomplete stage from `last.pt` and skips stages already completed under the same output root.

Individual stages:

```bash
python train.py --stage 1 --config configs/colab.yaml
python train.py --stage 2 --config configs/colab.yaml
python train.py --stage 3 --config configs/colab.yaml
```

Checkpoints are written atomically as `best.pt`, `last.pt`, and `epoch_XXX.pt`. Each best model is also exported as compact `best.safetensors` and `best_fp16.safetensors` files. Stage 2 loads Stage 1's `best.pt`; Stage 3 loads Stage 2's `best.pt`. Metrics are appended to `metrics.jsonl`.

Natural validation uses global TP/FP/FN counts over the whole validation set and sweeps the configurable thresholds in `common.validation_thresholds`. The curve F1 from the best threshold selects the checkpoint instead of relying on a fixed 0.5 cutoff or an average of per-batch F1 values.

The default config enables deterministic mode. On the same hardware/software stack, an interrupted run can restore optimizer, scheduler, scaler, and all recorded RNG state:

```bash
python train.py \
  --stage 1 \
  --config configs/colab.yaml \
  --resume outputs/stage1/last.pt
```

A completed `last.pt` copied into a new output directory is also accepted; the compact best exports are recreated if no epochs remain.

## Inference and export

```bash
python infer.py --checkpoint outputs/stage3/best.pt --input example.png
python export.py --checkpoint outputs/stage3/best.pt --output outputs/teed_curves.onnx
```

See [NOTICE.md](NOTICE.md) for upstream attribution and dataset terms.
