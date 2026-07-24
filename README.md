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

Stage 1 automatically downloads the approximately 250 KB public TEED checkpoint from `fal/teed` on Hugging Face unless `common.teed_checkpoint` is set. It refuses to continue if too few compatible weights load.

## Model size

The default model has **67,980 parameters**. The TEED-compatible shared encoder and edge branch contain **58,910 parameters**; the curve decoder and curve-context block add **9,070 parameters**.

A validated export from the default architecture is approximately:

- `best.safetensors`: 277,808 bytes (FP32)
- `best_fp16.safetensors`: 141,840 bytes (FP16)

The resumable `.pt` files are larger because they also contain optimizer, scheduler, scaler, configuration, and RNG state.

## Fast verification

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test.py
pytest -q  # includes a miniature Stage 1 -> 2 -> 3 checkpoint-handoff run
```

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

Natural curve pseudo-labels are generated conservatively from annotated edges by measuring tangent change along traced contours. Synthetic curve targets are exact.

CurveML support accepts point-set files stored as `.csv` or `.csv.xz`. The manifest builder prioritizes `point_set_clean`/`point_cloud_clean` files so metadata CSVs and perturbed outlier sequences are not accidentally rendered as connected curves. Clone the official repository or copy an existing dataset directory:

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

This downloads public BIPED data through KaggleHub, clones the BIDS BSDS500 mirror, derives natural curve pseudo-labels, and optionally clones CurveML. Dataset files are never committed to this repository.

## Train

```bash
python train_all.py --config configs/colab.yaml
```

Individual stages:

```bash
python train.py --stage 1 --config configs/colab.yaml
python train.py --stage 2 --config configs/colab.yaml
python train.py --stage 3 --config configs/colab.yaml
```

Checkpoints are written atomically as `best.pt` and `last.pt`. Each best model is also exported as compact `best.safetensors` and `best_fp16.safetensors` files. Stage 2 loads Stage 1's `best.pt`; Stage 3 loads Stage 2's `best.pt`. Metrics are appended to `metrics.jsonl`.

## Inference and export

```bash
python infer.py --checkpoint outputs/stage3/best.pt --input example.png
python export.py --checkpoint outputs/stage3/best.pt --output outputs/teed_curves.onnx
```

See [COLAB.md](COLAB.md) for a copy-paste Colab flow and [NOTICE.md](NOTICE.md) for upstream attribution and dataset terms.
