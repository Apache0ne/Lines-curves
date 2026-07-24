# Google Colab

Use a GPU runtime. The commands are intentionally minimal; all training logic is in Python.

```python
import os, subprocess, sys

repo = "/content/Lines-curves"
if not os.path.exists(repo):
    subprocess.run(["git", "clone", "https://github.com/Apache0ne/Lines-curves.git", repo], check=True)
os.chdir(repo)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
subprocess.run([sys.executable, "scripts/smoke_test.py"], check=True)
```

Download and normalize BIPED v2 and BSDS500 automatically. Add `--with-curveml` to clone the official CurveML repository and build a point-set manifest:

```python
import subprocess, sys
subprocess.run([
    sys.executable, "scripts/colab_setup.py",
    "--with-curveml",
], check=True)
```

The CurveML clone is optional because the exact procedural generator is always available. Stage 2 automatically uses CurveML CSV point sets whenever the manifest contains them.

Run the full curriculum:

```python
import subprocess, sys
subprocess.run([sys.executable, "train_all.py", "--config", "configs/colab.yaml"], check=True)
```

The final checkpoint is:

```text
/content/Lines-curves/outputs/stage3/best.pt             # resumable training checkpoint
/content/Lines-curves/outputs/stage3/best.safetensors    # compact FP32 weights
/content/Lines-curves/outputs/stage3/best_fp16.safetensors
```

For Google Drive persistence, edit `common.output_root` in `configs/colab.yaml` to a mounted Drive directory before training.
