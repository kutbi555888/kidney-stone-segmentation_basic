
# ============================================================
# KIDNEY STONE SEGMENTATION - FINAL EVALUATION
#
# Bu skript final modelni TEST datasetda baholaydi.
#
# Hisoblanadigan asosiy metrikalar:
#
# 1. Mean per-image Dice
# 2. Mean per-image IoU
# 3. Mean per-image Precision
# 4. Mean per-image Recall
#
# 5. Global Dice
# 6. Global IoU
# 7. Global Precision
# 8. Global Recall
#
# 9. Mean absolute area error (pixel)
#
# MUHIM:
# - threshold validation set orqali tanlangan.
# - min_component_pixels validation set orqali tanlangan.
# - TEST set parametr tanlash uchun ishlatilmaydi.
#
# KSSD2025 fizik pixel spacing bermagani sabab:
# mm² yoki mm³ hisoblanmaydi.
#
# "Stone count" metric olib tashlangan.
# Connected componentlar bu datasetda klinik stone count
# sifatida ishonchli emas.
# ============================================================


import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scipy import ndimage
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ============================================================
# 1-BO'LIM:
# src papkani Python pathga qo'shamiz
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR)
    )


from common import (
    load_config,
    resolve_path
)

from dataset import KidneyStoneDataset

from model import build_model


# ============================================================
# 2-BO'LIM:
# Kichik connected componentlarni olib tashlash
#
# Faqat PREDICTION maskga qo'llanadi.
# Ground Truth mask o'zgartirilmaydi.
# ============================================================

def remove_small_components(
    mask,
    min_pixels
):

    mask = mask.astype(
        bool
    )

    labeled, n_components = ndimage.label(
        mask
    )

    cleaned = np.zeros_like(
        mask,
        dtype=np.uint8
    )


    for component_id in range(
        1,
        n_components + 1
    ):

        component = (
            labeled == component_id
        )

        area = int(
            component.sum()
        )


        if area >= min_pixels:

            cleaned[
                component
            ] = 1


    return cleaned


# ============================================================
# 3-BO'LIM:
# Bitta image uchun TP, FP, FN asosida metric hisoblash
# ============================================================

def calculate_metrics(
    target,
    prediction
):

    target = target.astype(
        bool
    )

    prediction = prediction.astype(
        bool
    )


    tp = int(
        np.logical_and(
            prediction,
            target
        ).sum()
    )


    fp = int(
        np.logical_and(
            prediction,
            np.logical_not(target)
        ).sum()
    )


    fn = int(
        np.logical_and(
            np.logical_not(prediction),
            target
        ).sum()
    )


    eps = 1e-7


    dice = (
        2 * tp + eps
    ) / (
        2 * tp
        + fp
        + fn
        + eps
    )


    iou = (
        tp + eps
    ) / (
        tp
        + fp
        + fn
        + eps
    )


    precision = (
        tp + eps
    ) / (
        tp
        + fp
        + eps
    )


    recall = (
        tp + eps
    ) / (
        tp
        + fn
        + eps
    )


    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


# ============================================================
# 4-BO'LIM:
# Asosiy evaluation funksiyasi
# ============================================================

def evaluate(
    config_path
):

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    cfg = load_config(
        config_path
    )


    threshold = float(
        cfg["training"]["threshold"]
    )


    min_component_pixels = int(
        cfg["postprocess"][
            "min_component_pixels"
        ]
    )


    image_size = tuple(
        int(x)
        for x in cfg["data"]["image_size"]
    )


    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )

    print(
        "Threshold:",
        threshold
    )

    print(
        "Min component pixels:",
        min_component_pixels
    )


    # ========================================================
    # 5-BO'LIM:
    # Test splitni yuklash
    # ========================================================

    splits_dir = resolve_path(
        cfg["data"]["splits_dir"]
    )


    test_csv_path = (
        splits_dir
        / "test.csv"
    )


    test_df = pd.read_csv(
        test_csv_path
    )


    print(
        "Test images:",
        len(test_df)
    )


    # ========================================================
    # 6-BO'LIM:
    # Dataset va DataLoader
    #
    # batch_size=1:
    # har bir image uchun alohida metric yozish oson bo'ladi.
    # ========================================================

    test_dataset = KidneyStoneDataset(
        test_df,
        image_size=image_size,
        augment=False
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(
            cfg["training"].get(
                "num_workers",
                2
            )
        ),
        pin_memory=(
            device.type == "cuda"
        )
    )


    # ========================================================
    # 7-BO'LIM:
    # Final best modelni yuklash
    # ========================================================

    model = build_model(
        cfg
    ).to(
        device
    )


    model_path = resolve_path(
        cfg["paths"]["best_model"]
    )


    checkpoint = torch.load(
        model_path,
        map_location=device
    )


    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()


    print(
        "Model:",
        model_path
    )


    # ========================================================
    # 8-BO'LIM:
    # Test prediction
    # ========================================================

    rows = []


    # Global metric uchun yig'iladi
    total_tp = 0
    total_fp = 0
    total_fn = 0


    with torch.no_grad():

        for index, batch in enumerate(
            tqdm(
                test_loader,
                desc="Test"
            )
        ):

            image = batch[
                "image"
            ].to(
                device,
                non_blocking=True
            )


            target = (
                batch["mask"][0, 0]
                .cpu()
                .numpy()
                > 0.5
            ).astype(
                np.uint8
            )


            # ------------------------------------------------
            # Model inference
            # ------------------------------------------------

            logits = model(
                image
            )


            probability = torch.sigmoid(
                logits
            )[0, 0].cpu().numpy()


            # ------------------------------------------------
            # Final threshold
            # ------------------------------------------------

            prediction = (
                probability
                >= threshold
            ).astype(
                np.uint8
            )


            # ------------------------------------------------
            # Final postprocessing
            # ------------------------------------------------

            prediction = remove_small_components(
                prediction,
                min_component_pixels
            )


            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                target,
                prediction
            )


            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]


            # ------------------------------------------------
            # Pixel area
            # ------------------------------------------------

            true_area = int(
                target.sum()
            )


            predicted_area = int(
                prediction.sum()
            )


            area_abs_error = abs(
                predicted_area
                - true_area
            )


            # ------------------------------------------------
            # Test CSV tartibi DataLoader bilan bir xil
            # ------------------------------------------------

            image_path = str(
                test_df.iloc[index][
                    "image"
                ]
            )


            mask_path = str(
                test_df.iloc[index][
                    "mask"
                ]
            )


            rows.append(
                {
                    "image": image_path,
                    "mask": mask_path,

                    "dice":
                        metrics["dice"],

                    "iou":
                        metrics["iou"],

                    "precision":
                        metrics["precision"],

                    "recall":
                        metrics["recall"],

                    "true_area_pixels":
                        true_area,

                    "predicted_area_pixels":
                        predicted_area,

                    "area_abs_error_pixels":
                        area_abs_error
                }
            )


    # ========================================================
    # 9-BO'LIM:
    # Per-image natijalarni DataFrame qilish
    # ========================================================

    results_df = pd.DataFrame(
        rows
    )


    # ========================================================
    # 10-BO'LIM:
    # Mean per-image metrics
    # ========================================================

    mean_dice = float(
        results_df[
            "dice"
        ].mean()
    )


    mean_iou = float(
        results_df[
            "iou"
        ].mean()
    )


    mean_precision = float(
        results_df[
            "precision"
        ].mean()
    )


    mean_recall = float(
        results_df[
            "recall"
        ].mean()
    )


    mean_area_abs_error = float(
        results_df[
            "area_abs_error_pixels"
        ].mean()
    )


    # ========================================================
    # 11-BO'LIM:
    # Global / micro metrics
    #
    # Barcha test pixellar bitta katta mask sifatida qaraladi.
    # ========================================================

    eps = 1e-7


    global_dice = (
        2 * total_tp + eps
    ) / (
        2 * total_tp
        + total_fp
        + total_fn
        + eps
    )


    global_iou = (
        total_tp + eps
    ) / (
        total_tp
        + total_fp
        + total_fn
        + eps
    )


    global_precision = (
        total_tp + eps
    ) / (
        total_tp
        + total_fp
        + eps
    )


    global_recall = (
        total_tp + eps
    ) / (
        total_tp
        + total_fn
        + eps
    )


    # ========================================================
    # 12-BO'LIM:
    # Final summary
    # ========================================================

    summary = {

        "n_test":
            int(len(results_df)),

        # ----------------------------------------------------
        # Mean per-image
        # ----------------------------------------------------

        "mean_dice":
            mean_dice,

        "mean_iou":
            mean_iou,

        "mean_precision":
            mean_precision,

        "mean_recall":
            mean_recall,

        # ----------------------------------------------------
        # Global / micro
        # ----------------------------------------------------

        "global_dice":
            float(global_dice),

        "global_iou":
            float(global_iou),

        "global_precision":
            float(global_precision),

        "global_recall":
            float(global_recall),

        # ----------------------------------------------------
        # Area
        # ----------------------------------------------------

        "mean_area_abs_error_pixels":
            mean_area_abs_error,

        # ----------------------------------------------------
        # Final pipeline parametrlar
        # ----------------------------------------------------

        "threshold":
            threshold,

        "min_component_pixels":
            min_component_pixels,

        # ----------------------------------------------------
        # Project limitation
        # ----------------------------------------------------

        "important_note": (
            "KSSD2025 2D TIF pipeline fizik pixel spacing "
            "ishlatmagani sabab area pixel birlikda beriladi; "
            "mm2 yoki mm3 hisoblanmaydi. Connected-component "
            "soni klinik kidney-stone count sifatida "
            "baholanmaydi."
        )
    }


    # ========================================================
    # 13-BO'LIM:
    # Natijalarni saqlash
    # ========================================================

    test_metrics_path = resolve_path(
        cfg["paths"]["test_csv"]
    )


    test_summary_path = resolve_path(
        cfg["paths"]["test_summary"]
    )


    test_metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    test_summary_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    results_df.to_csv(
        test_metrics_path,
        index=False
    )


    with open(
        test_summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )


    # ========================================================
    # 14-BO'LIM:
    # Terminalga final natijalarni chiqarish
    # ========================================================

    print(
        "\n"
        "========================================"
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "========================================"
    )


    print(
        f"N test              : "
        f"{summary['n_test']}"
    )


    print(
        "\n--- Mean per-image metrics ---"
    )

    print(
        f"Mean Dice           : "
        f"{mean_dice:.4f}"
    )

    print(
        f"Mean IoU            : "
        f"{mean_iou:.4f}"
    )

    print(
        f"Mean Precision      : "
        f"{mean_precision:.4f}"
    )

    print(
        f"Mean Recall         : "
        f"{mean_recall:.4f}"
    )


    print(
        "\n--- Global metrics ---"
    )

    print(
        f"Global Dice         : "
        f"{global_dice:.4f}"
    )

    print(
        f"Global IoU          : "
        f"{global_iou:.4f}"
    )

    print(
        f"Global Precision    : "
        f"{global_precision:.4f}"
    )

    print(
        f"Global Recall       : "
        f"{global_recall:.4f}"
    )


    print(
        "\n--- Area ---"
    )

    print(
        f"Mean area MAE       : "
        f"{mean_area_abs_error:.2f} pixels"
    )


    print(
        "\n--- Final pipeline ---"
    )

    print(
        f"Threshold           : "
        f"{threshold}"
    )

    print(
        f"Min component pixels: "
        f"{min_component_pixels}"
    )


    print(
        "\nSaved:"
    )

    print(
        test_metrics_path
    )

    print(
        test_summary_path
    )


    print(
        "\nNOTE:"
    )

    print(
        summary[
            "important_note"
        ]
    )


    return summary


# ============================================================
# 15-BO'LIM:
# Terminal entry point
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "KSSD2025 Kidney Stone "
            "Segmentation Final Evaluation"
        )
    )


    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml"
    )


    args = parser.parse_args()


    evaluate(
        args.config
    )


if __name__ == "__main__":
    main()
