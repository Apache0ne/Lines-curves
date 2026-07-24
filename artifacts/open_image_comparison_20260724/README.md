# Base TEED vs Stage 3 FP32 — open-image comparison

This benchmark uses the exact uploaded Base TEED checkpoint and exact current Stage 3 FP32 pilot on 24 inputs: 17 distinct open sample images from `skimage.data` plus seven detailed crops.

- Base SHA256: `d0109e7f40e7d9f1f495d34947eb08167e8fbb0a13b4e6ab3121261fb8d5a416`
- Stage 3 SHA256: `132ff6716fdacbc5845d575d5e025e8ef46ec608a823d16f3d341be93941762c`
- Model parameters: `67,980`
- Inference maximum side: `512 px`
- Curve-overlay threshold: `0.65`

These photographs do not include edge/curve ground truth, so this is a visual comparison and output-statistics report—not an F1 or accuracy benchmark.

## Compact comparison sheets

![Sheet 1](comparison_sheet_01.webp)

![Sheet 2](comparison_sheet_02.webp)

![Sheet 3](comparison_sheet_03.webp)

![Sheet 4](comparison_sheet_04.webp)

## Inputs

The distinct images are: astronaut, cameraman, coffee, coins, Chelsea the cat, motion-blurred clock, SpaceX rocket launch, horse silhouette, Hubble deep field, printed text, brick, grass, gravel, microscopy cell, immunohistochemistry, retinal microaneurysms, and human retina. Seven additional crops test fine curved structures and local detail.

The corresponding `skimage.data` function documentation states CC0, public-domain, or no-known-copyright-restrictions status for the originals. Full-resolution PNG sheets and exact metrics were generated in the execution workspace; `metrics.json` in this directory contains the per-image output statistics.
