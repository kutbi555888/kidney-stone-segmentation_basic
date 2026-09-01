
"""
Kidney stone kabi juda kichik foreground uchun loss funksiyasi.

Bu datasetda stone pixel juda kam (~0.1% atrofida).
Oddiy BCE backgroundni juda kuchli o'rganib qolishi mumkin.

Shuning uchun:
1. Weighted BCE
2. Tversky Loss

birgalikda ishlatiladi.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor
) -> torch.Tensor:
    """
    Funksiya nomi eski train.py bilan moslik uchun
    dice_bce_loss bo'lib qoldi.

    Lekin ichida:
        Weighted BCE + Tversky Loss
    ishlatiladi.
    """

    # ==========================================================
    # 1-BO'LIM: Weighted BCE
    #
    # Stone pixel juda kam.
    # pos_weight > 1 stone pixellarga ko'proq ahamiyat beradi.
    #
    # Datasetdagi nazariy inverse ratio ~700+ bo'lishi mumkin,
    # lekin juda katta weight trainingni beqaror qilishi mumkin.
    # Shu sabab konservativ 50 dan boshlaymiz.
    # ==========================================================

    pos_weight = torch.tensor(
        [50.0],
        dtype=logits.dtype,
        device=logits.device
    ).view(1, 1, 1)

    weighted_bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight
    )

    # ==========================================================
    # 2-BO'LIM: Tversky Loss
    #
    # beta > alpha bo'lgani uchun False Negative,
    # ya'ni haqiqiy stone'ni o'tkazib yuborishga
    # ko'proq penalty beriladi.
    # ==========================================================

    probs = torch.sigmoid(logits)

    dims = (0, 2, 3)

    true_positive = (
        probs * targets
    ).sum(dim=dims)

    false_positive = (
        probs * (1.0 - targets)
    ).sum(dim=dims)

    false_negative = (
        (1.0 - probs) * targets
    ).sum(dim=dims)

    alpha = 0.3
    beta = 0.7
    smooth = 1.0

    tversky = (
        true_positive + smooth
    ) / (
        true_positive
        + alpha * false_positive
        + beta * false_negative
        + smooth
    )

    tversky_loss = 1.0 - tversky.mean()

    # ==========================================================
    # 3-BO'LIM: Ikkala lossni birlashtirish
    # ==========================================================

    total_loss = (
        0.4 * weighted_bce
        +
        0.6 * tversky_loss
    )

    return total_loss
