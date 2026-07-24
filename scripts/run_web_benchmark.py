from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from lines_curves.model import TEEDCurves

MEAN_BGR = np.asarray([103.939, 116.779, 123.68], dtype=np.float32)


def font(size: int, bold: bool = False):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def decode_stage3(parts_dir: Path, output: Path) -> None:
    chunks = sorted(parts_dir.glob("stage3_fp32.pth.gz.b64.part*"))
    if not chunks:
        raise FileNotFoundError(f"No Stage 3 checkpoint chunks under {parts_dir}")
    encoded = "".join(path.read_text(encoding="ascii").strip() for path in chunks)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path, retries: int = 4) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 1024:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "Lines-curves-benchmark/1.0 (GitHub Actions)"})
    error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            if len(data) < 1024:
                raise RuntimeError(f"Downloaded file is unexpectedly small: {len(data)} bytes")
            output.write_bytes(data)
            return
        except Exception as exc:
            error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url}: {error}")


def commons_download_url(filename: str) -> str:
    return "https://commons.wikimedia.org/wiki/Special:Redirect/file/" + urllib.parse.quote(filename, safe="")


def load_models(base_path: Path, stage3_path: Path):
    base = TEEDCurves()
    report = base.load_teed_checkpoint(base_path)
    if report["loaded_keys"] != 36:
        raise RuntimeError(f"Base TEED load mismatch: {report}")
    base.eval()

    stage3 = TEEDCurves()
    state = torch.load(stage3_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    stage3.load_state_dict(state, strict=True)
    stage3.eval()
    return base, stage3, report


def model_input(image_bgr: np.ndarray, max_side: int = 768):
    h0, w0 = image_bgr.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))
    h = max(8, int(round(h0 * scale / 8)) * 8)
    w = max(8, int(round(w0 * scale / 8)) * 8)
    resized = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    tensor = torch.from_numpy((resized.astype(np.float32) - MEAN_BGR).transpose(2, 0, 1))[None]
    return tensor, (h0, w0)


def infer(model: TEEDCurves, tensor: torch.Tensor, original_hw: tuple[int, int]):
    with torch.inference_mode():
        output = model(tensor)
    result = {}
    for key in ("edge", "curve"):
        prob = torch.sigmoid(output[key])[0, 0].cpu().numpy()
        prob = cv2.resize(prob, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_CUBIC)
        result[key] = np.clip(prob, 0.0, 1.0)
    return result


def heatmap(prob: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(np.clip(prob * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def overlay(image: np.ndarray, prob: np.ndarray, threshold: float, color=(0, 255, 255)) -> np.ndarray:
    result = image.copy()
    mask = prob >= threshold
    layer = np.zeros_like(result)
    layer[:] = color
    result[mask] = cv2.addWeighted(result, 0.35, layer, 0.65, 0)[mask]
    return result


def letterbox_bgr(image: np.ndarray, size: tuple[int, int], background=(24, 24, 24)) -> Image.Image:
    target_w, target_h = size
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), background)
    canvas.paste(pil, ((target_w - pil.width) // 2, (target_h - pil.height) // 2))
    return canvas


def draw_text_fit(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], width: int, fnt, fill="white"):
    words = text.split()
    lines, current = [], ""
    for word in words:
        trial = (current + " " + word).strip()
        if draw.textbbox((0, 0), trial, font=fnt)[2] <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    y = xy[1]
    for line in lines[:2]:
        draw.text((xy[0], y), line, font=fnt, fill=fill)
        y += fnt.size + 2 if hasattr(fnt, "size") else 14


def make_sheet(records, output: Path, title: str, cell=(300, 220)):
    columns = ["Source", "Base TEED edge", "Stage 3 edge", "Stage 3 curve", "Stage 3 curve overlay"]
    header_h, row_label_h, gap = 76, 42, 8
    width = len(columns) * cell[0] + (len(columns) + 1) * gap
    row_h = row_label_h + cell[1] + gap
    height = header_h + len(records) * row_h + gap
    canvas = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), title, font=font(28, True), fill="white")
    for col, label in enumerate(columns):
        x = gap + col * (cell[0] + gap)
        draw.text((x + 4, 48), label, font=font(16, True), fill=(210, 210, 215))
    y = header_h
    for index, record in enumerate(records):
        bg = (28, 28, 32) if index % 2 == 0 else (23, 23, 27)
        draw.rectangle((0, y, width, y + row_h), fill=bg)
        label = f"{record['id']} — {record['title']} [{record['license']}]"
        draw_text_fit(draw, label, (gap + 4, y + 6), width - 2 * gap, font(16, True), fill="white")
        tiles = [record["source"], record["base_edge"], record["stage3_edge"], record["stage3_curve"], record["curve_overlay"]]
        for col, tile in enumerate(tiles):
            x = gap + col * (cell[0] + gap)
            canvas.paste(letterbox_bgr(tile, cell), (x, y + row_label_h))
        y += row_h
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="benchmarks/web_sources.json")
    parser.add_argument("--checkpoint-parts", default="artifacts/checkpoints")
    parser.add_argument("--output", default="artifacts/web_benchmark_20260724")
    parser.add_argument("--base-url", default="https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/7_model.pth")
    parser.add_argument("--base-sha256", default="d0109e7f40e7d9f1f495d34947eb08167e8fbb0a13b4e6ab3121261fb8d5a416")
    args = parser.parse_args()

    output = Path(args.output)
    work = output / "_work"
    work.mkdir(parents=True, exist_ok=True)
    base_path = work / "7_model.pth"
    stage3_path = work / "stage3_fp32.pth"
    download(args.base_url, base_path)
    if sha256(base_path) != args.base_sha256:
        raise RuntimeError(f"Base checkpoint SHA mismatch: {sha256(base_path)}")
    decode_stage3(Path(args.checkpoint_parts), stage3_path)
    stage3_sha = sha256(stage3_path)
    if stage3_sha != "132ff6716fdacbc5845d575d5e025e8ef46ec608a823d16f3d341be93941762c":
        raise RuntimeError(f"Stage 3 checkpoint SHA mismatch: {stage3_sha}")

    base, stage3, load_report = load_models(base_path, stage3_path)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    records = []
    metrics = []
    source_dir = work / "sources"
    for item in manifest["images"]:
        ext = Path(item["filename"]).suffix or ".jpg"
        local = source_dir / f"{item['id']}{ext}"
        image_url = item.get("download_url") or commons_download_url(item["filename"])
        download(image_url, local)
        image = cv2.imread(str(local), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read {local}")
        tensor, original_hw = model_input(image)
        base_out = infer(base, tensor, original_hw)
        stage_out = infer(stage3, tensor, original_hw)
        base_edge = heatmap(base_out["edge"])
        stage_edge = heatmap(stage_out["edge"])
        stage_curve = heatmap(stage_out["curve"])
        curve_overlay = overlay(image, stage_out["curve"], 0.65, color=(0, 255, 255))
        diff = np.abs(stage_out["edge"] - base_out["edge"])
        metric = {
            **{k: item[k] for k in ("id", "title", "category", "license", "source_page")},
            "download_url": image_url,
            "width": int(image.shape[1]), "height": int(image.shape[0]),
            "base_edge_mean": float(base_out["edge"].mean()),
            "stage3_edge_mean": float(stage_out["edge"].mean()),
            "stage3_curve_mean": float(stage_out["curve"].mean()),
            "base_edge_coverage_t045": float((base_out["edge"] >= 0.45).mean()),
            "stage3_edge_coverage_t060": float((stage_out["edge"] >= 0.60).mean()),
            "stage3_curve_coverage_t065": float((stage_out["curve"] >= 0.65).mean()),
            "edge_mean_absolute_change": float(diff.mean()),
        }
        metrics.append(metric)
        records.append({**item, "source": image, "base_edge": base_edge, "stage3_edge": stage_edge, "stage3_curve": stage_curve, "curve_overlay": curve_overlay})
        print(f"PROCESSED={item['id']} {image.shape[1]}x{image.shape[0]}")

    for sheet_index, start in enumerate(range(0, len(records), 8), 1):
        make_sheet(records[start:start+8], output / f"comparison_sheet_{sheet_index:02d}.png", f"Base TEED vs Stage 3 FP32 — open web images ({start+1}–{min(start+8, len(records))} of {len(records)})")
    make_sheet(records, output / "comparison_sheet_all_24.png", f"Base TEED vs Stage 3 FP32 — all {len(records)} open web images", cell=(250, 184))

    (output / "metrics.json").write_text(json.dumps({"base_sha256": sha256(base_path), "stage3_sha256": stage3_sha, "load_report": load_report, "images": metrics}, indent=2), encoding="utf-8")
    with (output / "sources_and_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    readme = [
        "# Base TEED vs Stage 3 FP32: open-web benchmark", "",
        f"This benchmark runs the uploaded base `7_model.pth` and the current Stage 3 FP32 pilot on {len(records)} freely reusable Wikimedia Commons images.", "",
        f"- Base SHA256: `{sha256(base_path)}`", f"- Stage 3 SHA256: `{stage3_sha}`", "- Inference maximum side: 768 px", "- Sheet curve threshold: 0.65", "",
        "## Comparison sheets", "",
    ]
    for i in range(1, math.ceil(len(records)/8)+1):
        readme += [f"![Comparison sheet {i}](comparison_sheet_{i:02d}.png)", ""]
    readme += ["## Source and license manifest", "", "| # | Image | Category | License |", "|---:|---|---|---|"]
    for i, item in enumerate(manifest["images"], 1):
        readme.append(f"| {i} | [{item['title']}]({item['source_page']}) | {item['category']} | {item['license']} |")
    readme += ["", "`metrics.json` and `sources_and_metrics.csv` contain per-image probability coverage and Base-vs-Stage-3 edge-change measurements."]
    (output / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    import shutil
    shutil.rmtree(work)
    print(f"BENCHMARK_COMPLETE={output}")


if __name__ == "__main__":
    main()
