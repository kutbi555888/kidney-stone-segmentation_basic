
"""
2D CT image va binary kidney-stone maskni
PyTorch tensoriga aylantiradigan Dataset.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset


# ==============================================================
# 1-BO'LIM:
# TIF tasvirni grayscale numpy array sifatida o'qish.
# ==============================================================

def _load_gray(path: str) -> np.ndarray:

    arr = np.asarray(
        Image.open(path)
    )

    # Agar image RGB bo'lsa grayscale qilamiz.
    if arr.ndim == 3:

        arr = (
            arr[..., :3]
            .astype(np.float32)
            .mean(axis=2)
        )

    return arr.astype(np.float32)


# ==============================================================
# 2-BO'LIM:
# CT intensity qiymatlarini 0..1 oralig'iga normalize qilish.
#
# Extreme qiymatlar ta'sirini kamaytirish uchun
# 0.5 va 99.5 percentile ishlatiladi.
# ==============================================================

def _normalize_ct(
    arr: np.ndarray
) -> np.ndarray:

    lo, hi = np.percentile(
        arr,
        [0.5, 99.5]
    )

    if hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max())

    if hi <= lo:

        return np.zeros_like(
            arr,
            dtype=np.float32
        )

    arr = np.clip(
        arr,
        lo,
        hi
    )

    arr = (
        arr - lo
    ) / (
        hi - lo
    )

    return arr.astype(
        np.float32
    )


# ==============================================================
# 3-BO'LIM:
# PyTorch Dataset.
# ==============================================================

class KidneyStoneDataset(Dataset):

    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: tuple[int, int],
        augment: bool = False
    ):

        self.frame = frame.reset_index(
            drop=True
        )

        self.image_size = image_size

        self.augment = augment


    def __len__(self):

        return len(
            self.frame
        )


    def __getitem__(
        self,
        idx: int
    ):

        row = self.frame.iloc[idx]

        # ======================================================
        # 4-BO'LIM:
        # CT image o'qiladi va normalize qilinadi.
        # ======================================================

        image = _normalize_ct(
            _load_gray(
                row["image"]
            )
        )

        # ======================================================
        # 5-BO'LIM:
        # Ground-truth mask binary formatga o'tkaziladi.
        #
        # 0 = background
        # 1 = kidney stone
        # ======================================================

        mask = _load_gray(
            row["mask"]
        )

        mask = (
            mask > 0
        ).astype(
            np.float32
        )

        # [H,W] -> [1,H,W]

        image_t = torch.from_numpy(
            image
        )[None, ...]

        mask_t = torch.from_numpy(
            mask
        )[None, ...]

        # ======================================================
        # 6-BO'LIM:
        # 512x512 resolution.
        #
        # IMAGE:
        # bilinear interpolation
        #
        # MASK:
        # nearest interpolation
        #
        # Mask uchun nearest juda muhim,
        # chunki binary label buzilmasligi kerak.
        # ======================================================

        image_t = F.interpolate(
            image_t[None],
            size=self.image_size,
            mode="bilinear",
            align_corners=False
        )[0]

        mask_t = F.interpolate(
            mask_t[None],
            size=self.image_size,
            mode="nearest"
        )[0]

        # ======================================================
        # 7-BO'LIM:
        # Training augmentation.
        #
        # Hozir faqat left-right flip ishlatiladi.
        #
        # Vertical flip va 90° rotation olib tashlandi,
        # chunki axial CT anatomiyasini ortiqcha buzishi mumkin.
        # ======================================================

        if self.augment:

            if random.random() < 0.5:

                image_t = torch.flip(
                    image_t,
                    dims=[2]
                )

                mask_t = torch.flip(
                    mask_t,
                    dims=[2]
                )

        return {

            "image":
                image_t.contiguous(),

            "mask":
                mask_t.contiguous(),

            "image_path":
                row["image"],

            "mask_path":
                row["mask"],
        }
