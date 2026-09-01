"""
Datasetni topish, image-mask juftlash, tekshirish va train/val/test split yaratish.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from common import load_config, resolve_path, save_json, set_seed
from data_utils import discover_pairs, save_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    raw_dir = resolve_path(cfg["data"]["raw_dir"])

    if not raw_dir.exists():
        raise RuntimeError(f"Dataset papkasi topilmadi: {raw_dir}. Avval src/download_data.py ni run qiling.")

    print("1-BO'LIM: image-mask juftlarini qidirish")
    df = discover_pairs(raw_dir)
    if df.empty:
        raise RuntimeError(
            "Image-mask juftligi topilmadi. Dataset yuklangan bo'lsa, data/raw/kssd2025 ichidagi "
            "birinchi 30 fayl pathini yuboring; pairing qoidasi real strukturaga moslanadi."
        )

    print("Topilgan juftliklar:", len(df))

    print("\n2-BO'LIM: masklarda haqiqiy foreground borligini tekshirish")
    positives = []
    for _, row in df.iterrows():
        mask = np.asarray(Image.open(row["mask"]))
        if mask.ndim == 3:
            mask = mask[..., :3].mean(axis=2)
        positives.append(int((mask > 0).sum()))
    df["stone_pixels"] = positives
    print("Stone pixel > 0 bo'lgan masklar:", int((df["stone_pixels"] > 0).sum()), "/", len(df))
    print("Median stone pixels:", float(df["stone_pixels"].median()))

    manifest_path = resolve_path(cfg["data"]["manifest_csv"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)

    print("\n3-BO'LIM: image-level train/val/test split")
    # KSSD2025 public TIF paketida patient metadata aniq bo'lmasa, image-level split qilinadi.
    # Agar keyin patient ID mavjudligi aniqlansa, group splitga o'tish kerak.
    seed = int(cfg["project"]["seed"])
    test_size = float(cfg["data"]["test_size"])
    val_size = float(cfg["data"]["val_size"])

    train_val, test = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
    val_relative = val_size / (1.0 - test_size)
    train, val = train_test_split(train_val, test_size=val_relative, random_state=seed, shuffle=True)

    splits_dir = resolve_path(cfg["data"]["splits_dir"])
    save_split(train, splits_dir / "train.csv")
    save_split(val, splits_dir / "val.csv")
    save_split(test, splits_dir / "test.csv")

    summary = {
        "total_pairs": int(len(df)),
        "train": int(len(train)),
        "val": int(len(val)),
        "test": int(len(test)),
        "positive_masks": int((df["stone_pixels"] > 0).sum()),
        "median_stone_pixels": float(df["stone_pixels"].median()),
        "split_level": "image-level",
        "note": "Agar patient ID mavjud bo'lsa, keyinchalik patient/group-level split tavsiya qilinadi."
    }
    save_json(summary, splits_dir / "dataset_summary.json")
    print(summary)


if __name__ == "__main__":
    main()
