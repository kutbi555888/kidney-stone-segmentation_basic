"""Binary kidney-stone segmentation metrikalari va post-processing."""
from __future__ import annotations

import numpy as np
from scipy import ndimage

EPS = 1e-7


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    precision = (tp + EPS) / (tp + fp + EPS)
    recall = (tp + EPS) / (tp + fn + EPS)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def clean_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, n = ndimage.label(mask.astype(bool))
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n + 1):
        comp = labels == i
        if int(comp.sum()) >= min_pixels:
            out[comp] = 1
    return out


def component_stats(mask: np.ndarray, min_pixels: int = 1) -> list[dict]:
    clean = clean_components(mask, min_pixels)
    labels, n = ndimage.label(clean)
    stats = []
    for i in range(1, n + 1):
        ys, xs = np.where(labels == i)
        if len(xs) == 0:
            continue
        area = int(len(xs))
        stats.append({
            "stone_id": i,
            "area_pixels": area,
            "bbox_width_pixels": int(xs.max() - xs.min() + 1),
            "bbox_height_pixels": int(ys.max() - ys.min() + 1),
            "centroid_x": float(xs.mean()),
            "centroid_y": float(ys.mean()),
        })
    return stats
