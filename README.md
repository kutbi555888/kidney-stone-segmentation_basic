# Kidney Stone 2D Segmentation — KSSD2025

## Project haqida

Bu project axial CT tasvirlarida buyrak toshi (kidney stone) hududini **2D semantic segmentation** yordamida ajratish uchun tayyorlangan.

Asosiy maqsad faqat “tosh bormi?” deb classification qilish emas. Model toshga tegishli pixel hududini mask ko‘rinishida ajratadi. Shu maskdan keyin:

- kidney-stone hududining pixel maydoni;
- alohida segmentlangan komponentlar soni;
- maskning shakli va bounding o‘lchamlari;
- Dice, IoU, Precision va Recall

hisoblanadi.

> Muhim: KSSD2025 2D TIF tasvirlardan iborat. Shu sababli bu projectda 3D volume (mm³) hisoblanmaydi. Fizik pixel spacing mavjudligi tasdiqlanmaguncha natijalar pixel birliklarida beriladi.

## Segmentation turi

Project **binary 2D semantic segmentation** vazifasini bajaradi:

- `0` — background / kidney-stone bo‘lmagan pixel;
- `1` — kidney-stone pixeli.

## Dataset

Project **KSSD2025 — Kidney Stone Segmentation Dataset** uchun tayyorlangan.

Dataset xususiyatlari:

- 838 ta annotated axial CT image;
- binary kidney-stone segmentation mask;
- TIF format;
- kidney-stone holatlariga yo‘naltirilgan;
- KSSD2025 mualliflari original CT Kidney Dataset asosida mask annotation tayyorlagan.

Kaggle dataset nomi:

`murillobouzon/kssd2025-kidney-stone-segmentation-dataset`

Dataset project ichidagi `src/download_data.py` orqali KaggleHub yordamida yuklanadi.

## Model

Asosiy baseline model:

**2D U-Net (MONAI + PyTorch)**

Input:

```text
1-channel CT image
```

Output:

```text
1-channel kidney-stone probability mask
```

Loss:

```text
Dice Loss + Binary Cross Entropy
```

Default training konfiguratsiyasi vaqtni juda cho‘zmaslik uchun yengil qilingan:

- image size: `256 × 256`;
- batch size: `8`;
- epoch: `12`;
- AMP: yoqilgan;
- early stopping patience: `3`.

## Project nimalardan tashkil topgan

```text
kidney_stone_project_kssd2025/
├── app/
├── configs/
├── data/
├── models/
├── notebooks/
├── results/
├── src/
├── run_pipeline.py
├── requirements.txt
├── README.md
└── PROJECT_STRUCTURE.md
```

Batafsil izoh `PROJECT_STRUCTURE.md` faylida berilgan.

## Ish jarayoni

```text
Kaggle KSSD2025
      ↓
Dataset download
      ↓
Image-mask pairing
      ↓
Dataset inspection
      ↓
Train / Validation / Test split
      ↓
2D preprocessing + augmentation
      ↓
2D U-Net training
      ↓
Best checkpoint
      ↓
Test evaluation
      ↓
Dice / IoU / Precision / Recall
      ↓
Stone count + pixel-area
      ↓
Single-image prediction
      ↓
Gradio app
```

## Run qilish tartibi

### 1. Kutubxonalarni o‘rnatish

```bash
pip install -r requirements.txt
```

### 2. Muhitni tekshirish

```bash
python src/check_environment.py
```

### 3. Datasetni yuklash

```bash
python src/download_data.py --config configs/config.yaml
```

### 4. Datasetni tayyorlash

```bash
python src/prepare_data.py --config configs/config.yaml
```

Bu bosqich:

- image va masklarni topadi;
- ularni juftlaydi;
- maskda foreground borligini tekshiradi;
- `manifest.csv` yaratadi;
- train/validation/test split yaratadi.

### 5. Sample visualization

```bash
python src/visualize_sample.py --config configs/config.yaml
```

Natija:

```text
results/figures/sample_overlay.png
```

### 6. Modelni train qilish

```bash
python src/train.py --config configs/config.yaml
```

Training uzilib qolgan bo‘lsa:

```bash
python src/train.py --config configs/config.yaml --resume
```

Best model:

```text
models/best_2d_unet.pth
```

### 7. Test evaluation

```bash
python src/evaluate.py --config configs/config.yaml
```

Natijalar:

```text
results/metrics/test_metrics.csv
results/metrics/test_summary.json
```

### 8. Bitta image uchun prediction

```bash
python src/predict.py path/to/image.tif --config configs/config.yaml
```

Natijalar `results/predictions/` ichiga saqlanadi.

### 9. Gradio app

```bash
python app/app.py
```

## Evaluation metrics

Project quyidagilarni hisoblaydi:

- Dice;
- IoU;
- Precision;
- Recall;
- predicted stone count;
- true stone count;
- segmented area (pixels);
- area absolute error (pixels).

## Muhim cheklovlar

1. Dataset 2D CT slicelardan iborat, shuning uchun haqiqiy 3D volume hisoblanmaydi.
2. Fizik pixel spacing tasdiqlanmaguncha `mm`, `mm²`, `mm³` natijalari chiqarilmaydi.
3. Public paketda patient ID/group metadata aniq mavjud bo‘lmasa, split image-level bo‘ladi. Agar patient ID topilsa, group/patient-level splitga o‘tish kerak.
4. Project klinik tashxis vositasi emas; ta’lim va research maqsadidagi AI pipeline.

## Keyingi rivojlantirish variantlari

- 512×512 training;
- U-Net++;
- Attention U-Net;
- TransUNet;
- YOLO-Seg baseline;
- k-fold cross-validation;
- threshold tuning;
- external validation;
- patient-level split (agar patient ID mavjud bo‘lsa).
