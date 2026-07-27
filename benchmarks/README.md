# Benchmark records

`synthetic_192_seed1234.json` is generated with:

```bash
python scripts/benchmark.py \
  --size 192 \
  --seed 1234 \
  --tolerance 2.5 \
  --output outputs/benchmark
```

The benchmark uses tolerance-aware boundary matching. It is designed for regression testing and controlled corruption analysis. It is not an official BSDS500, BIPED, NYUDv2, or Multicue score.
