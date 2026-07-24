# Google Colab

Use a GPU runtime. The repository includes a complete notebook: [`Lines_Curves_Training_Colab.ipynb`](Lines_Curves_Training_Colab.ipynb).

## 1. Clone, install, and run repository tests

```python
from pathlib import Path
import os, subprocess, sys

repo = Path("/content/Lines-curves")
if repo.exists():
    subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"], check=True)
else:
    subprocess.run(["git", "clone", "https://github.com/Apache0ne/Lines-curves.git", str(repo)], check=True)
os.chdir(repo)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "compileall", "-q", "."], check=True)
subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
subprocess.run([sys.executable, "scripts/smoke_test.py"], check=True)
```

## 2. Download and normalize data

Add `--with-curveml` to clone CurveML and build a clean point-set manifest:

```python
subprocess.run([
    sys.executable, "scripts/colab_setup.py",
    "--with-curveml",
], check=True)
```

The CurveML clone is optional because the exact procedural generator is always available. Stage 2 uses clean CurveML point sets from `.csv` or `.csv.xz` files whenever the manifest contains them.

## 3. Validate the complete setup

```python
subprocess.run([
    sys.executable, "scripts/preflight.py",
    "--config", "configs/colab.yaml",
    "--download-teed",
    "--report", "outputs/preflight.json",
], check=True)
```

Do not start training unless this prints `PREFLIGHT_STATUS=PASS`.

## 4. Run the full curriculum

```python
subprocess.run([sys.executable, "train_all.py", "--config", "configs/colab.yaml"], check=True)
```

The final files are:

```text
/content/Lines-curves/outputs/stage3/best.pt
/content/Lines-curves/outputs/stage3/best.safetensors
/content/Lines-curves/outputs/stage3/best_fp16.safetensors
```

Every stage also keeps `last.pt`, `epoch_XXX.pt`, and `metrics.jsonl`.

## Google Drive persistence

Mount Drive and change `common.output_root` before preflight/training. The included notebook performs this edit automatically when `USE_GOOGLE_DRIVE = True`.

## Resume an interrupted stage

```python
subprocess.run([
    sys.executable, "train.py",
    "--stage", "1",
    "--config", "configs/colab.yaml",
    "--resume", "outputs/stage1/last.pt",
], check=True)
```

The checkpoint restores optimizer, scheduler, AMP scaler, and recorded Python/NumPy/Torch/CUDA RNG state. Exact tensor-for-tensor replay is validated on the same deterministic software/hardware stack; different CUDA/PyTorch versions can still change floating-point results.
