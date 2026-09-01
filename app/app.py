
# ============================================================
# KIDNEY STONE SEGMENTATION - GRADIO FINAL APP
#
# Bu dastur:
# 1. CT TIF rasmni qabul qiladi
# 2. Final 2D U-Net model orqali segmentation qiladi
# 3. threshold = config.yaml dagi final qiymatni ishlatadi
# 4. Kichik connected componentlarni olib tashlaydi
# 5. CT, probability map, mask va overlayni ko'rsatadi
# 6. Stone pixel area va connected regionlarni chiqaradi
#
# MUHIM:
# - Model KSSD2025 2D TIF datasetida train qilingan.
# - Fizik pixel spacing mavjud emas.
# - Shu sabab mm² yoki mm³ hisoblanmaydi.
# - Bu dastur klinik tashxis vositasi emas.
# ============================================================

import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from PIL import Image
from scipy import ndimage


# ============================================================
# 1-BO'LIM:
# Project pathlarini sozlash
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_DIR = (
    PROJECT_ROOT / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR)
)

# Relative pathlar har doim project rootdan ishlashi uchun
os.chdir(
    PROJECT_ROOT
)


# ============================================================
# 2-BO'LIM:
# Project funksiyalarini import qilish
# ============================================================

from common import (
    load_config,
    resolve_path
)

from model import build_model

from predict import (
    preprocess_image,
    remove_small_components
)


# ============================================================
# 3-BO'LIM:
# Configni yuklash
# ============================================================

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "config.yaml"
)

cfg = load_config(
    str(CONFIG_PATH)
)


IMAGE_SIZE = tuple(
    int(x)
    for x in cfg["data"]["image_size"]
)

THRESHOLD = float(
    cfg["training"]["threshold"]
)

MIN_COMPONENT_PIXELS = int(
    cfg["postprocess"][
        "min_component_pixels"
    ]
)


# ============================================================
# 4-BO'LIM:
# GPU / CPU aniqlash
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# 5-BO'LIM:
# Modelni FAQAT BIR MARTA yuklash
#
# Har predictionda modelni qayta yuklamaymiz.
# Shu sabab Gradio tezroq ishlaydi.
# ============================================================

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
    "Model yuklandi:",
    model_path
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Min component pixels:",
    MIN_COMPONENT_PIXELS
)


# ============================================================
# 6-BO'LIM:
# Float image -> uint8
# ============================================================

def to_uint8(image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.clip(
        image,
        0,
        1
    )

    return (
        image * 255
    ).astype(
        np.uint8
    )


# ============================================================
# 7-BO'LIM:
# Overlay yaratish
#
# Predicted stone region qizil overlay bilan ko'rsatiladi.
# ============================================================

def create_overlay(
    image,
    mask
):

    base = to_uint8(
        image
    )

    # Grayscale -> RGB
    rgb = np.stack(
        [
            base,
            base,
            base
        ],
        axis=-1
    ).astype(
        np.float32
    )


    mask_bool = (
        mask > 0
    )


    # Stone region uchun qizil layer
    red_layer = np.zeros_like(
        rgb
    )

    red_layer[
        ...,
        0
    ] = 255


    # Faqat mask bor joyda overlay
    rgb[
        mask_bool
    ] = (
        0.55
        * rgb[
            mask_bool
        ]
        +
        0.45
        * red_layer[
            mask_bool
        ]
    )


    return np.clip(
        rgb,
        0,
        255
    ).astype(
        np.uint8
    )


# ============================================================
# 8-BO'LIM:
# Connected region statistikasi
# ============================================================

def get_region_stats(
    mask
):

    labeled, n_regions = (
        ndimage.label(
            mask
        )
    )

    region_areas = []


    for component_id in range(
        1,
        n_regions + 1
    ):

        area = int(
            (
                labeled
                == component_id
            ).sum()
        )

        region_areas.append(
            area
        )


    region_areas = sorted(
        region_areas,
        reverse=True
    )


    return (
        int(n_regions),
        region_areas
    )


# ============================================================
# 9-BO'LIM:
# Asosiy Gradio prediction funksiyasi
# ============================================================

def segment_kidney_stone(
    image_path
):

    # --------------------------------------------------------
    # Upload qilinmagan bo'lsa
    # --------------------------------------------------------

    if image_path is None:

        return (
            None,
            None,
            None,
            None,
            "Rasm yuklanmadi."
        )


    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    try:

        (
            tensor,
            original_normalized,
            resized
        ) = preprocess_image(
            image_path,
            IMAGE_SIZE
        )

    except Exception as e:

        return (
            None,
            None,
            None,
            None,
            f"Rasmni o'qishda xato: {e}"
        )


    tensor = tensor.to(
        device
    )


    # --------------------------------------------------------
    # Model inference
    # --------------------------------------------------------

    try:

        with torch.no_grad():

            logits = model(
                tensor
            )

            probability = torch.sigmoid(
                logits
            )[0, 0].cpu().numpy()

    except Exception as e:

        return (
            None,
            None,
            None,
            None,
            f"Prediction xatosi: {e}"
        )


    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    raw_mask = (
        probability
        >= THRESHOLD
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # Final postprocessing
    # --------------------------------------------------------

    final_mask = remove_small_components(
        raw_mask,
        MIN_COMPONENT_PIXELS
    )


    # --------------------------------------------------------
    # Stone area
    # --------------------------------------------------------

    stone_area = int(
        final_mask.sum()
    )


    # --------------------------------------------------------
    # Connected regions
    # --------------------------------------------------------

    (
        n_regions,
        region_areas
    ) = get_region_stats(
        final_mask
    )


    # --------------------------------------------------------
    # Visualizationlar
    # --------------------------------------------------------

    ct_image = to_uint8(
        resized
    )


    probability_image = (
        np.clip(
            probability,
            0,
            1
        )
        * 255
    ).astype(
        np.uint8
    )


    mask_image = (
        final_mask
        * 255
    ).astype(
        np.uint8
    )


    overlay_image = create_overlay(
        resized,
        final_mask
    )


    # --------------------------------------------------------
    # Natija matni
    # --------------------------------------------------------

    if stone_area == 0:

        detection_text = (
            "Final threshold va postprocessingdan "
            "keyin segmentlangan region topilmadi."
        )

    else:

        detection_text = (
            "Segmentlangan kidney-stone region "
            "mavjud."
        )


    regions_text = (
        ", ".join(
            str(x)
            for x in region_areas
        )
        if region_areas
        else "0"
    )


    result_text = f"""
### Segmentation natijasi

**Holat:** {detection_text}

**Predicted stone area:** `{stone_area}` pixel

**Estimated connected regions:** `{n_regions}`

**Region areas:** `{regions_text}` pixel

**Threshold:** `{THRESHOLD}`

**Minimum component size:** `{MIN_COMPONENT_PIXELS}` pixel

---

**Eslatma:** `Estimated connected regions` klinik jihatdan
aniq kidney-stone soni degani emas.

KSSD2025 pipeline fizik pixel spacing ishlatmagani sabab
maydon `mm²` emas, **pixel** birlikda beriladi.

Bu model tadqiqot va ta'lim maqsadida yaratilgan,
klinik tashxis o'rnini bosmaydi.
"""


    return (
        ct_image,
        probability_image,
        mask_image,
        overlay_image,
        result_text
    )


# ============================================================
# 10-BO'LIM:
# Gradio interfeys
# ============================================================

with gr.Blocks(
    title="Kidney Stone Segmentation"
) as demo:


    gr.Markdown(
        """
# Kidney Stone Segmentation

2D CT tasvirlarda kidney-stone hududini
U-Net yordamida segmentatsiya qilish.

**Final model:** 2D U-Net  
**Dataset:** KSSD2025  
**Input:** 512 × 512  
**Threshold:** 0.993  
**Postprocessing:** `< 50 pixel` componentlar olib tashlanadi
"""
    )


    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                type="filepath",
                label="CT TIF rasmni yuklang"
            )

            predict_button = gr.Button(
                "Segmentation qilish",
                variant="primary"
            )

            clear_button = gr.ClearButton(
                value="Tozalash"
            )


        with gr.Column():

            result_text = gr.Markdown(
                value=(
                    "CT rasm yuklang va "
                    "**Segmentation qilish** "
                    "tugmasini bosing."
                )
            )


    gr.Markdown(
        "## Visualization"
    )


    with gr.Row():

        ct_output = gr.Image(
            label="Preprocessed CT"
        )

        probability_output = gr.Image(
            label="Probability map"
        )


    with gr.Row():

        mask_output = gr.Image(
            label="Final segmentation mask"
        )

        overlay_output = gr.Image(
            label="Stone overlay"
        )


    # ========================================================
    # Button action
    # ========================================================

    predict_button.click(
        fn=segment_kidney_stone,
        inputs=[
            input_image
        ],
        outputs=[
            ct_output,
            probability_output,
            mask_output,
            overlay_output,
            result_text
        ]
    )


    clear_button.add(
        [
            input_image,
            ct_output,
            probability_output,
            mask_output,
            overlay_output,
            result_text
        ]
    )


# ============================================================
# 11-BO'LIM:
# Appni ishga tushirish
#
# share=True:
# Google Colab uchun public Gradio link beradi.
# ============================================================

if __name__ == "__main__":

    demo.launch(
        share=True,
        debug=True
    )
