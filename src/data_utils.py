"""KSSD2025 image-mask juftlarini topish va manifest bilan ishlash."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MASK_WORDS = {"mask", "masks", "label", "labels", "seg", "segmentation", "groundtruth", "ground_truth", "gt"}
IMAGE_WORDS = {"image", "images", "img", "imgs", "scan", "scans", "ct"}


def _is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def _tokens(p: Path) -> set[str]:
    text = "/".join(part.lower() for part in p.parts)
    return set(re.split(r"[^a-z0-9]+", text))


def _looks_like_mask(p: Path) -> bool:
    return bool(_tokens(p) & MASK_WORDS)


def _looks_like_image(p: Path) -> bool:
    return bool(_tokens(p) & IMAGE_WORDS) and not _looks_like_mask(p)


def _normalize_stem(stem: str) -> str:
    s = stem.lower()
    suffixes = [
        "_mask", "-mask", " mask", "_masks", "_label", "-label", "_labels",
        "_seg", "-seg", "_segmentation", "-segmentation", "_gt", "-gt",
        "_image", "-image", "_img", "-img", "_ct", "-ct"
    ]
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if s.endswith(suf):
                s = s[: -len(suf)]
                changed = True
    return re.sub(r"[^a-z0-9]+", "", s)


def _relative_context_key(p: Path) -> str:
    """Bir xil filename turli splitlarda bo'lsa split kontekstini ham keyga qo'shadi."""
    stem = _normalize_stem(p.stem)
    parent_tokens = [x.lower() for x in p.parts[:-1] if x.lower() not in MASK_WORDS | IMAGE_WORDS]
    tail = parent_tokens[-2:] if parent_tokens else []
    return "__".join(tail + [stem])


def discover_pairs(root: Path) -> pd.DataFrame:
    """
    Dataset strukturasi turlicha bo'lsa ham image va masklarni juftlashga harakat qiladi.

    1) avval images/ va masks/ kabi papka nomlarini ishlatadi;
    2) keyin filename suffixlari (_mask, _label...) bo'yicha fallback qiladi.
    """
    all_files = sorted([p for p in root.rglob("*") if _is_image_file(p)])
    if not all_files:
        return pd.DataFrame(columns=["image", "mask"])

    masks = [p for p in all_files if _looks_like_mask(p)]
    images = [p for p in all_files if _looks_like_image(p)]

    # Agar image papka nomlari aniq bo'lmasa, mask bo'lmaganlarning hammasini image candidate deb olamiz.
    if not images:
        images = [p for p in all_files if p not in masks]

    mask_by_key: dict[str, list[Path]] = {}
    mask_by_stem: dict[str, list[Path]] = {}
    for m in masks:
        mask_by_key.setdefault(_relative_context_key(m), []).append(m)
        mask_by_stem.setdefault(_normalize_stem(m.stem), []).append(m)

    pairs: list[dict[str, str]] = []
    used_masks: set[Path] = set()

    for img in images:
        key = _relative_context_key(img)
        stem_key = _normalize_stem(img.stem)
        candidates = mask_by_key.get(key, [])
        if len(candidates) != 1:
            candidates = mask_by_stem.get(stem_key, [])

        if len(candidates) == 1:
            mask = candidates[0]
            pairs.append({"image": str(img.resolve()), "mask": str(mask.resolve())})
            used_masks.add(mask)

    # Fallback: barcha fayllar orasida bir xil normalized stem bo'yicha image-mask qidirish.
    if not pairs and masks:
        non_masks = [p for p in all_files if p not in masks]
        for img in non_masks:
            candidates = mask_by_stem.get(_normalize_stem(img.stem), [])
            if len(candidates) == 1:
                pairs.append({"image": str(img.resolve()), "mask": str(candidates[0].resolve())})

    df = pd.DataFrame(pairs).drop_duplicates() if pairs else pd.DataFrame(columns=["image", "mask"])
    return df


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
