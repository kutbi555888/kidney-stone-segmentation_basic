
# ============================================================
# KIDNEY STONE SEGMENTATION - FINAL PREDICTION
#
# Bu skript yangi 2D CT TIF rasmga prediction qiladi.
#
# Final pipeline:
#   1. CT rasmni o'qish
#   2. Trainingdagi preprocessingni qo'llash
#   3. 2D U-Net prediction
#   4. Sigmoid probability
#   5. Threshold = config.yaml
#   6. Kichik connected componentlarni olib tashlash
#   7. Mask, overlay va probability mapni saqlash
#
# MUHIM:
# KSSD2025 datasetida fizik pixel spacing ishlatilmaydi.
# Shu sabab area pixel birlikda hisoblanadi.
# ============================================================

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage

sys.path.append(str(Path(__file__).resolve().parent))

from common import load_config, resolve_path
from model import build_model


# ============================================================
# 1-BO'LIM:
# Rasmni preprocessing qilish
#
# Trainingdagi dataset.py bilan mos:
# - grayscale
# - 0.5 / 99.5 percentile clipping
# - 0..1 normalization
# - configdagi image_size ga resize
# ============================================================

def preprocess_image(
    image_path,
    image_size
):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Rasm topilmadi: {image_path}"
        )

    # --------------------------------------------------------
    # TIF rasmni ochamiz
    # --------------------------------------------------------

    image = np.asarray(
        Image.open(image_path)
    )


    # --------------------------------------------------------
    # Agar RGB bo'lsa grayscale qilamiz
    # --------------------------------------------------------

    if image.ndim == 3:

        image = image[
            ..., :3
        ].mean(axis=2)


    image = image.astype(
        np.float32
    )


    # --------------------------------------------------------
    # Percentile clipping
    # Training preprocessing bilan bir xil.
    # --------------------------------------------------------

    low = np.percentile(
        image,
        0.5
    )

    high = np.percentile(
        image,
        99.5
    )


    if high > low:

        image = np.clip(
            image,
            low,
            high
        )

        image = (
            image - low
        ) / (
            high - low
        )

    else:

        image = np.zeros_like(
            image,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # PIL resize uchun 0..255 ga o'tkazamiz
    # --------------------------------------------------------

    image_uint8 = (
        image * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )


    # image_size = [512, 512]
    height = int(
        image_size[0]
    )

    width = int(
        image_size[1]
    )


    resized = Image.fromarray(
        image_uint8
    ).resize(
        (width, height),
        resample=Image.Resampling.BILINEAR
    )


    resized = np.asarray(
        resized
    ).astype(
        np.float32
    ) / 255.0


    # --------------------------------------------------------
    # PyTorch format:
    #
    # H × W
    #   ↓
    # 1 × H × W
    #   ↓
    # batch uchun
    # 1 × 1 × H × W
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        resized
    ).float()

    tensor = tensor.unsqueeze(
        0
    ).unsqueeze(
        0
    )


    return (
        tensor,
        image,
        resized
    )


# ============================================================
# 2-BO'LIM:
# Kichik connected componentlarni olib tashlash
# ============================================================

def remove_small_components(
    mask,
    min_pixels
):

    mask = mask.astype(
        bool
    )

    labeled, n = ndimage.label(
        mask
    )

    cleaned = np.zeros_like(
        mask,
        dtype=np.uint8
    )


    for component_id in range(
        1,
        n + 1
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
# Final prediction funksiyasi
# ============================================================

def predict_image(
    image_path,
    config_path
):

    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    cfg = load_config(
        config_path
    )


    image_size = tuple(
        int(x)
        for x in cfg["data"]["image_size"]
    )


    threshold = float(
        cfg["training"]["threshold"]
    )


    min_component_pixels = int(
        cfg["postprocess"][
            "min_component_pixels"
        ]
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


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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

    print(
        "Threshold:",
        threshold
    )

    print(
        "Min component pixels:",
        min_component_pixels
    )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    (
        tensor,
        original_normalized,
        resized
    ) = preprocess_image(
        image_path,
        image_size
    )


    tensor = tensor.to(
        device
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            tensor
        )

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    raw_mask = (
        probability
        >= threshold
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # Postprocessing
    # --------------------------------------------------------

    final_mask = remove_small_components(
        raw_mask,
        min_component_pixels
    )


    # --------------------------------------------------------
    # Quantitative analysis
    # --------------------------------------------------------

    predicted_area_pixels = int(
        final_mask.sum()
    )


    labeled, n_regions = ndimage.label(
        final_mask
    )


    region_areas = []


    for component_id in range(
        1,
        n_regions + 1
    ):

        area = int(
            (
                labeled == component_id
            ).sum()
        )

        region_areas.append(
            area
        )


    region_areas = sorted(
        region_areas,
        reverse=True
    )


    result = {
        "probability": probability,
        "raw_mask": raw_mask,
        "final_mask": final_mask,
        "predicted_area_pixels":
            predicted_area_pixels,
        "estimated_connected_regions":
            int(n_regions),
        "region_areas_pixels":
            region_areas,
        "threshold":
            threshold,
        "min_component_pixels":
            min_component_pixels,
        "resized_image":
            resized
    }


    return result


# ============================================================
# 4-BO'LIM:
# Prediction natijalarini saqlash
# ============================================================

def save_prediction(
    image_path,
    result,
    output_dir
):

    image_path = Path(
        image_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    stem = image_path.stem


    # --------------------------------------------------------
    # Final binary mask
    # --------------------------------------------------------

    mask_path = (
        output_dir
        /
        f"{stem}_mask.png"
    )


    Image.fromarray(
        (
            result["final_mask"]
            * 255
        ).astype(
            np.uint8
        )
    ).save(
        mask_path
    )


    # --------------------------------------------------------
    # Probability map
    # --------------------------------------------------------

    probability_path = (
        output_dir
        /
        f"{stem}_probability.png"
    )


    probability_uint8 = (
        result["probability"]
        * 255
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )


    Image.fromarray(
        probability_uint8
    ).save(
        probability_path
    )


    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    figure_path = (
        output_dir
        /
        f"{stem}_prediction.png"
    )


    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 5)
    )


    # Original
    axes[0].imshow(
        result["resized_image"],
        cmap="gray"
    )

    axes[0].set_title(
        "CT"
    )


    # Probability
    im = axes[1].imshow(
        result["probability"],
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        "Probability"
    )

    plt.colorbar(
        im,
        ax=axes[1],
        fraction=0.046,
        pad=0.04
    )


    # Final mask
    axes[2].imshow(
        result["final_mask"],
        cmap="gray"
    )

    axes[2].set_title(
        "Final mask"
    )


    # Overlay
    axes[3].imshow(
        result["resized_image"],
        cmap="gray"
    )


    overlay = np.ma.masked_where(
        result["final_mask"] == 0,
        result["final_mask"]
    )


    axes[3].imshow(
        overlay,
        alpha=0.50
    )


    axes[3].set_title(
        "Stone overlay"
    )


    for ax in axes:
        ax.axis(
            "off"
        )


    plt.tight_layout()

    plt.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(
        fig
    )


    return {
        "mask_path": mask_path,
        "probability_path":
            probability_path,
        "figure_path":
            figure_path
    }


# ============================================================
# 5-BO'LIM:
# Terminal orqali ishga tushirish
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "KSSD2025 Kidney Stone "
            "Segmentation Prediction"
        )
    )


    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml"
    )


    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Prediction qilinadigan TIF rasm"
    )


    parser.add_argument(
        "--output_dir",
        type=str,
        default=None
    )


    args = parser.parse_args()


    cfg = load_config(
        args.config
    )


    if args.output_dir is None:

        output_dir = resolve_path(
            cfg["paths"]["prediction_dir"]
        )

    else:

        output_dir = Path(
            args.output_dir
        )


    result = predict_image(
        args.image,
        args.config
    )


    files = save_prediction(
        args.image,
        result,
        output_dir
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n===== PREDICTION RESULT ====="
    )

    print(
        "Predicted stone area:",
        result["predicted_area_pixels"],
        "pixels"
    )


    print(
        "Estimated connected regions:",
        result[
            "estimated_connected_regions"
        ]
    )


    print(
        "Region areas:",
        result[
            "region_areas_pixels"
        ]
    )


    print(
        "\nMask:",
        files["mask_path"]
    )


    print(
        "Probability:",
        files["probability_path"]
    )


    print(
        "Figure:",
        files["figure_path"]
    )


    print(
        "\nNOTE:"
    )

    print(
        "Area pixel birlikda berildi. "
        "KSSD2025 pipeline fizik pixel spacing "
        "ishlatmagani uchun mm² hisoblanmaydi."
    )


if __name__ == "__main__":
    main()
