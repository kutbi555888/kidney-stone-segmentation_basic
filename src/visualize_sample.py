"""Bitta CT image va ground-truth mask overlayini saqlaydi."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import load_config, resolve_path


def gray(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    val_path = resolve_path(cfg["data"]["splits_dir"]) / "val.csv"
    df = pd.read_csv(val_path)
    row = df.iloc[0]
    image = gray(row["image"])
    mask = gray(row["mask"]) > 0

    out = resolve_path(cfg["paths"]["sample_figure"])
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, cmap="gray")
    ax.imshow(np.ma.masked_where(~mask, mask), alpha=0.45)
    ax.set_title("CT + ground-truth stone mask")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saqlandi:", out)


if __name__ == "__main__":
    main()
