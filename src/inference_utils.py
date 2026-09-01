"""CLI va Gradio bir xil prediction kodidan foydalanishi uchun yordamchi funksiyalar."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import resolve_path
from metrics import clean_components, component_stats
from model import build_model


def load_gray_normalized(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(Image.open(path))
    if raw.ndim == 3:
        raw = raw[..., :3].astype(np.float32).mean(axis=2)
    raw = raw.astype(np.float32)
    lo, hi = np.percentile(raw, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(raw.min()), float(raw.max())
    norm = np.zeros_like(raw, dtype=np.float32) if hi <= lo else np.clip((raw - lo) / (hi - lo), 0, 1)
    return raw, norm


def load_trained_model(cfg: dict, device: torch.device):
    model = build_model(cfg).to(device)
    ckpt = torch.load(resolve_path(cfg["paths"]["best_model"]), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def predict_one(image_path: str | Path, cfg: dict, model=None, device=None, output_prefix="prediction") -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = load_trained_model(cfg, device)

    raw, norm = load_gray_normalized(image_path)
    orig_h, orig_w = norm.shape
    size = tuple(int(x) for x in cfg["data"]["image_size"])
    x = torch.from_numpy(norm)[None, None]
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False).to(device)

    with torch.no_grad():
        prob_small = torch.sigmoid(model(x))
    prob = F.interpolate(prob_small, size=(orig_h, orig_w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    threshold = float(cfg["training"]["threshold"])
    min_pixels = int(cfg["postprocess"]["min_component_pixels"])
    pred = clean_components(prob >= threshold, min_pixels)
    stones = component_stats(pred, min_pixels)

    out_dir = resolve_path(cfg["paths"]["prediction_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{output_prefix}_mask.png"
    overlay_path = out_dir / f"{output_prefix}_overlay.png"
    report_path = out_dir / f"{output_prefix}_report.json"

    Image.fromarray((pred * 255).astype(np.uint8)).save(mask_path)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(raw, cmap="gray")
    ax.imshow(np.ma.masked_where(pred == 0, pred), alpha=0.45)
    ax.set_title("Predicted kidney-stone mask")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = {
        "stone_detected": bool(len(stones) > 0),
        "stone_count": int(len(stones)),
        "total_segmented_area_pixels": int(pred.sum()),
        "image_area_percent": float(100.0 * pred.sum() / pred.size),
        "stones": stones,
        "note": "Bu 2D datasetda fizik pixel spacing berilmagani uchun mm/mm²/mm³ emas, pixel birliklari hisoblanadi."
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return {
        "report": report,
        "mask_path": str(mask_path),
        "overlay_path": str(overlay_path),
        "report_path": str(report_path),
    }
