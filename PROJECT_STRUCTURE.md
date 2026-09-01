# Project strukturasi va 0 dan bajarilgan ishlar

Bu hujjat project ichidagi har bir papka va kodning vazifasini o‘zbek tilida tushuntiradi.

## 1. `configs/`

### `configs/config.yaml`

Projectning asosiy sozlamalari shu faylda saqlanadi.

Unda:

- dataset path;
- train/validation/test foizlari;
- input image size;
- epoch;
- batch size;
- learning rate;
- model kanallari;
- threshold;
- checkpoint pathlari

berilgan.

Default training vaqtni haddan tashqari cho‘zmaslik uchun `12 epoch` qilingan.

---

## 2. `data/`

### `data/raw/`

Kaggle’dan yuklangan original KSSD2025 dataset shu yerga tushadi.

Default joy:

```text
data/raw/kssd2025/
```

### `data/processed/`

Datasetdan hosil qilingan manifest kabi oraliq metadata saqlanadi.

```text
data/processed/manifest.csv
```

### `data/splits/`

Train, validation va test splitlar:

```text
train.csv
val.csv
test.csv
dataset_summary.json
```

---

## 3. `src/`

### `src/check_environment.py`

Bu kod:

- Python versiyasini;
- PyTorch versiyasini;
- CUDA bor-yo‘qligini;
- GPU nomini;
- VRAM hajmini

tekshiradi.

### `src/download_data.py`

KSSD2025 datasetini KaggleHub orqali yuklaydi.

Asosiy vazifa:

```text
Kaggle
  ↓
KSSD2025
  ↓
data/raw/kssd2025/
```

### `src/data_utils.py`

Dataset ichidan image va mask fayllarni topadi va pairing qiladi.

Kod turli papka strukturalariga moslashishi uchun `images`, `masks`, `labels`, `_mask`, `_seg` kabi nomlarni aniqlaydi.

### `src/prepare_data.py`

Dataset preparationning asosiy kodi.

Bajaradigan ishlar:

1. image-mask juftlarini topish;
2. mask foreground pixelini tekshirish;
3. manifest yaratish;
4. train/validation/test split qilish;
5. dataset summary saqlash.

### `src/dataset.py`

PyTorch Dataset class.

Bajaradi:

- TIF/PNG/JPG image o‘qish;
- grayscale qilish;
- intensity normalization;
- resize;
- binary mask tayyorlash;
- training augmentation.

### `src/model.py`

MONAI asosidagi **2D U-Net** arxitekturasini yaratadi.

### `src/losses.py`

Kidney-stone foreground juda kichik bo‘lgani uchun:

```text
Dice Loss + BCE Loss
```

kombinatsiyasini hisoblaydi.

### `src/metrics.py`

Quyidagilarni hisoblaydi:

- Dice;
- IoU;
- Precision;
- Recall;
- connected-component stone count;
- component pixel area;
- bounding size.

### `src/train.py`

Model trainingning asosiy kodi.

Bajaradi:

1. train/val datasetni o‘qish;
2. DataLoader yaratish;
3. 2D U-Net yaratish;
4. AdamW optimizer;
5. mixed precision training;
6. validation;
7. best model saqlash;
8. checkpoint saqlash;
9. early stopping.

### `src/evaluate.py`

Best modelni test setda baholaydi.

Natijalar:

```text
results/metrics/test_metrics.csv
results/metrics/test_summary.json
```

### `src/visualize_sample.py`

Ground-truth maskni CT image ustiga overlay qilib ko‘rsatadi.

### `src/inference_utils.py`

`predict.py` va Gradio ikkalasi ishlatadigan umumiy inference funksiyalarini saqlaydi.

### `src/predict.py`

Bitta yangi CT image uchun:

```text
image
 ↓
2D U-Net
 ↓
probability mask
 ↓
binary mask
 ↓
connected components
 ↓
count + pixel-area + overlay
```

pipeline bajaradi.

---

## 4. `models/`

Training davomida:

```text
best_2d_unet.pth
last_checkpoint.pth
```

saqlanadi.

`best_2d_unet.pth` validation Dice bo‘yicha eng yaxshi model.

`last_checkpoint.pth` training uzilib qolsa davom ettirish uchun.

---

## 5. `results/`

### `results/figures/`

Sample va boshqa grafik natijalar.

### `results/metrics/`

Training history va test metrikalari.

### `results/predictions/`

Single-image predictiondan:

- mask;
- overlay;
- JSON report

saqlanadi.

---

## 6. `app/`

### `app/app.py`

Gradio interface.

Foydalanuvchi CT image yuklaydi va tizim:

- predicted stone mask;
- overlay;
- stone count;
- pixel-area;
- JSON report

chiqaradi.

---

## 7. `notebooks/`

### `notebooks/COLAB_RUN_KSSD2025.ipynb`

Projectni notebook orqali bosqichma-bosqich run qilish uchun yordamchi notebook.

Colab ishlatish bo‘yicha matnli qo‘llanma README ichiga kiritilmagan; notebook faqat run celllarni beradi.

---

## 8. `run_pipeline.py`

Download → prepare → visualization → train → evaluate bosqichlarini ketma-ket ishga tushirish uchun yordamchi script.

Masalan:

```bash
python run_pipeline.py
```

Dataset avvaldan mavjud bo‘lsa:

```bash
python run_pipeline.py --skip-download
```

---

# 0 dan boshlab projectda bajarilgan ishlar

## 1-bosqich — Muammo tanlandi

Axial CT image ichidagi kidney-stone hududini pixel darajasida segmentlash vazifasi tanlandi.

## 2-bosqich — Dataset tanlandi

Restricted 3D dataset o‘rniga ochiqroq va tez ishlash mumkin bo‘lgan KSSD2025 tanlandi.

## 3-bosqich — Vazifa 3D dan 2D ga o‘zgartirildi

Oldingi 3D NIfTI pipeline KSSD2025 uchun mos emasligi sababli:

```text
3D U-Net → 2D U-Net
NIfTI → TIF
3D volume → 2D pixel-area
```

o‘zgartirildi.

## 4-bosqich — Dataset downloader yaratildi

KaggleHub orqali datasetni olish kodi qo‘shildi.

## 5-bosqich — Pairing va manifest yaratildi

Image-mask fayllarni avtomatik topish va juftlash kodi yaratildi.

## 6-bosqich — Split yaratildi

Train/validation/test split pipeline yaratildi.

## 7-bosqich — 2D preprocessing yaratildi

Resize, normalization va augmentation qo‘shildi.

## 8-bosqich — 2D U-Net yaratildi

MONAI/PyTorch bilan binary semantic segmentation modeli qo‘shildi.

## 9-bosqich — Training pipeline yaratildi

AMP, checkpoint, best-model saving va early stopping qo‘shildi.

## 10-bosqich — Evaluation yaratildi

Dice, IoU, Precision, Recall, stone count va pixel-area metrikalari qo‘shildi.

## 11-bosqich — Prediction yaratildi

Bitta yangi CT image uchun mask, overlay va JSON report pipeline tayyorlandi.

## 12-bosqich — Gradio yaratildi

Projectning final demo interfeysi tayyorlandi.
