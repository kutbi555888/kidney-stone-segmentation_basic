# PROJECTGA IZOH — Kidney Stone Segmentation

## 1. Hujjat maqsadi

Ushbu `projectga_izoh.md` fayli **Kidney Stone Segmentation** loyihasini 0 dan oxirigacha batafsil tushuntiradi. Hujjatda loyiha vazifasi, dataset, preprocessing, model, loss, training, validation, threshold tuning, postprocessing, test, prediction, Gradio, papka strukturasi, har bir asosiy kod faylining vazifasi, run qilish bosqichlari, VS Code/Colab ishlatish tartibi, GitHub va limitationlar yoritiladi.

---

# 2. Loyiha nima haqida?

Bu loyiha 2D CT tasvirlarda **kidney stone (buyrak toshi)** joylashgan hududni avtomatik segmentatsiya qilish uchun ishlab chiqilgan.

Classification faqat:

```text
CT → stone bor / stone yo‘q
```

desa, bu loyiha semantic segmentation qiladi:

```text
CT image
   ↓
2D U-Net
   ↓
har bir pixel uchun stone ehtimoli
   ↓
binary segmentation mask
```

Final classlar:

```text
0 = Background
1 = Kidney Stone
```

Modelning asosiy vazifasi stone hududining aniq pixel maskini topish.

---

# 3. Final inference pipeline

```text
TIF CT image
    ↓
Grayscale
    ↓
0.5 / 99.5 percentile clipping
    ↓
0–1 normalization
    ↓
512 × 512 resize
    ↓
2D U-Net
    ↓
Logits
    ↓
Sigmoid probability map
    ↓
Threshold = 0.993
    ↓
Binary mask
    ↓
< 50 pixel connected componentlarni olib tashlash
    ↓
Final stone mask
    ↓
Pixel area + overlay + estimated connected regions
```

Muhim: `0.993` 99.3% klinik ishonch degani emas. Bu validation orqali tanlangan segmentation cutoff.

---

# 4. Dataset

Loyihada **KSSD2025 Kidney Stone Segmentation Dataset** ishlatilgan.

Asosiy statistika:

```text
838 ta 2D axial CT TIF image
838 ta corresponding binary label
```

Dataset strukturasi:

```text
data/raw/kssd2025/data/
├── image/
│   ├── 1.tif
│   ├── 10.tif
│   ├── 306.tif
│   ├── 312.tif
│   └── ...
└── label/
    ├── 1.tif
    ├── 10.tif
    ├── 306.tif
    ├── 312.tif
    └── ...
```

Misol:

```text
image/312.tif → original CT
label/312.tif → ground-truth mask
```

Barcha 838 maskda foreground mavjud bo‘lgan, ya’ni stone-negative case’lar bilan alohida test yetarli emas.

---

# 5. Nega 3D emas, 2D?

Dastlab 3D CT segmentation rejalashtirilgan edi. Restricted-access 3D dataset sabab ochiq KSSD2025 datasetga o‘tildi. Shu sabab final task 2D semantic segmentation bo‘ldi.

KSSD2025 pipeline fizik pixel spacing ishlatmagani uchun:

```text
area → pixels
```

bo‘yicha beriladi. `mm²`, `mm³`, haqiqiy 3D volume hisoblanmaydi.

---

# 6. Train / Validation / Test split

Final split:

```text
Total      : 838
Train      : 586
Validation : 126
Test       : 126
```

Patient ID mavjud bo‘lmagani sabab image-level split ishlatilgan. Agar bir patientga tegishli bir nechta slice mavjud bo‘lsa, patient-level leakage ehtimolini to‘liq inkor qilib bo‘lmaydi.

Validation quyidagilar uchun ishlatilgan:

- best checkpoint tanlash;
- threshold tuning;
- postprocessing tuning.

Test set faqat final evaluation uchun ishlatilgan.

---

# 7. Project strukturasi

```text
kidney_stone_project_2/
│
├── app/
│   └── app.py
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── kssd2025/
│   │       └── data/
│   │           ├── image/
│   │           └── label/
│   ├── processed/
│   │   └── manifest.csv
│   └── splits/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── models/
│   ├── best_2d_unet.pth
│   └── last_checkpoint.pth
│
├── notebooks/
│   └── COLAB_RUN_KSSD2025.ipynb
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── predictions/
│
├── src/
│   ├── check_environment.py
│   ├── common.py
│   ├── data_utils.py
│   ├── dataset.py
│   ├── download_data.py
│   ├── evaluate.py
│   ├── inference_utils.py
│   ├── inspect_data.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── predict.py
│   ├── prepare_data.py
│   ├── train.py
│   └── visualize_sample.py
│
├── .gitignore
├── README.md
├── PROJECT_STRUCTURE.md
├── projectga_izoh.md
├── requirements.txt
└── run_pipeline.py
```

---

# 8. `configs/config.yaml`

Bu projectning markaziy konfiguratsiya fayli.

Final muhim qiymatlar:

```text
image_size            = [512, 512]
epochs                = 25
batch_size            = 4
learning_rate         = 0.0002
weight_decay          = 0.00001
AMP                   = true
threshold             = 0.993
min_component_pixels  = 50
```

Model:

```text
in_channels  = 1
out_channels = 1
channels     = [16, 32, 64, 128]
strides      = [2, 2, 2]
```

`config.yaml`ning foydasi: train, evaluate, predict va Gradio bir xil parametrdan foydalanadi.

---

# 9. `data/raw/`

Original dataset shu yerda turadi. Full projectda dataset mavjud, portfolio versiyada esa GitHub uchun raw dataset olib tashlangan.

---

# 10. `data/processed/manifest.csv`

Image-mask pairing natijasi.

Asosiy ustunlar:

```text
image
mask
stone_pixels
```

`stone_pixels` ground-truth maskdagi foreground pixel soni.

---

# 11. `data/splits/`

```text
train.csv → 586 image
val.csv   → 126 image
test.csv  → 126 image
```

Har CSV image va unga mos mask pathini saqlaydi.

---

# 12. `models/`

## `best_2d_unet.pth`

Final model. Validation Dice bo‘yicha eng yaxshi checkpoint. Evaluation, prediction va Gradio shu modelni ishlatadi.

## `last_checkpoint.pth`

Trainingning oxirgi state’i. `--resume` uchun kerak. Best model bilan bir xil bo‘lishi shart emas.

---

# 13. `results/figures/`

Asosiy fayllar:

```text
sample_overlay.png
best_median_worst_cases.png
```

`sample_overlay.png` image-mask alignmentni tekshirish uchun.

`best_median_worst_cases.png` final error analysis uchun: 3 worst, 3 median, 3 best case.

---

# 14. `results/metrics/`

## `training_history.csv`

Har epoch uchun:

```text
epoch
train_loss
val_loss
val_dice
```

Final history 1–25 epochni o‘z ichiga oladi.

## `test_metrics.csv`

Har test image uchun:

```text
image
mask
dice
iou
precision
recall
true_area_pixels
predicted_area_pixels
area_abs_error_pixels
```

## `test_summary.json`

Final umumiy metrikalar.

---

# 15. `results/predictions/`

Prediction script yaratadigan fayllar:

```text
*_mask.png
*_probability.png
*_prediction.png
```

Masalan:

```text
312_mask.png
312_probability.png
312_prediction.png
```

---

# 16. `src/check_environment.py`

Environmentni tekshiradi:

- Python;
- PyTorch;
- CUDA;
- GPU;
- kerakli package’lar.

Run:

```bash
python src/check_environment.py
```

---

# 17. `src/common.py`

Umumiy utility funksiyalar:

- config o‘qish;
- project root aniqlash;
- relative pathni absolute pathga aylantirish;
- random seed;
- umumiy helperlar.

Ko‘p scriptlar `load_config()` va `resolve_path()`dan foydalanadi.

---

# 18. `src/data_utils.py`

Data processing uchun reusable yordamchi logic saqlanadigan modul. Image-mask pairing, path va metadata bilan ishlashni modular qilish uchun ishlatiladi.

---

# 19. `src/dataset.py`

PyTorch Dataset class.

Image preprocessing:

```text
TIF read
  ↓
grayscale
  ↓
float32
  ↓
0.5 / 99.5 percentile clipping
  ↓
0–1 normalization
  ↓
512 × 512 bilinear resize
  ↓
PyTorch tensor
```

Mask:

```text
TIF read
  ↓
mask > 0
  ↓
binary
  ↓
nearest-neighbor resize
  ↓
tensor
```

Training augmentation: konservativ horizontal flip.

Mask nearest-neighbor bilan resize qilinadi, chunki bilinear interpolation class qiymatlarini buzishi mumkin.

---

# 20. Nega 512 × 512?

Initial versiyada 256 × 256 ishlatilgan. Kidney-stone region juda kichik bo‘lgani sabab detail yo‘qolgan. Final 512 × 512 kichik foregroundni yaxshiroq saqlash uchun tanlangan.

---

# 21. `src/download_data.py`

KSSD2025 datasetni yuklab olish uchun.

Run:

```bash
python src/download_data.py --config configs/config.yaml
```

---

# 22. `src/inspect_data.py`

Datasetni dastlabki tekshiradi:

- image soni;
- mask soni;
- shape;
- pairing;
- mask qiymatlari;
- foreground mavjudligi.

---

# 23. `src/prepare_data.py`

Asosiy data preparation script.

```text
Raw files
  ↓
image qidirish
  ↓
matching label
  ↓
stone pixel count
  ↓
manifest.csv
  ↓
train / val / test CSV
```

Run:

```bash
python src/prepare_data.py --config configs/config.yaml
```

---

# 24. `src/visualize_sample.py`

Datasetdan sample image, mask va overlay yaratadi. Trainingdan oldin pairing va alignmentni ko‘z bilan tekshirish uchun ishlatiladi.

Run:

```bash
python src/visualize_sample.py --config configs/config.yaml
```

---

# 25. `src/model.py`

Final 2D U-Net modelini yaratadi.

```text
Input
 ↓
Encoder
 ↓
Bottleneck
 ↓
Decoder
 ↓
1-channel segmentation logits
```

Skip connectionlar encoder spatial featurelarini decoderga uzatadi.

Final channels:

```text
16 → 32 → 64 → 128
```

Stone kichik bo‘lgani uchun haddan tashqari downsampling ishlatilmagan.

---

# 26. Nega U-Net?

U-Net medical image segmentation uchun qulay:

- pixel-level output;
- skip connection;
- kichik datasetlarda ham foydali;
- localization yaxshi;
- arxitekturasi tushunarli.

---

# 27. `src/losses.py`

Initial BCE + Dice kichik foreground uchun yetarli bo‘lmagan.

Dataset foreground ratio taxminan:

```text
0.135%
```

Final loss:

```text
0.4 × Weighted BCE
+
0.6 × Tversky Loss
```

Parametrlar:

```text
pos_weight = 50
alpha = 0.3
beta  = 0.7
```

Weighted BCE foregroundga ko‘proq weight beradi. Tversky esa FP va FN balansini boshqaradi; `beta=0.7` false-negativega kattaroq e’tibor beradi.

---

# 28. `src/metrics.py`

Segmentation metric helperlari:

- Dice;
- IoU;
- Precision;
- Recall;
- connected component helperlari.

Formulalar:

```text
Dice      = 2TP / (2TP + FP + FN)
IoU       = TP / (TP + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```

---

# 29. `src/train.py`

Trainingning asosiy scripti.

```text
config
 ↓
train / val CSV
 ↓
Dataset + DataLoader
 ↓
2D U-Net
 ↓
Weighted BCE + Tversky
 ↓
optimizer
 ↓
training loop
 ↓
validation
 ↓
best checkpoint
```

Final config:

```text
epochs        = 25
batch_size    = 4
learning_rate = 0.0002
weight_decay  = 0.00001
AMP           = true
```

Run:

```bash
python src/train.py --config configs/config.yaml
```

Resume:

```bash
python src/train.py --config configs/config.yaml --resume
```

---

# 30. Initial model va muammo

Initial 256 × 256 modelda Dice juda past bo‘lgan:

```text
≈ 0.02
```

Sabablar:

- foreground juda kichik;
- class imbalance;
- 256 resize detailni kamaytirgan;
- initial loss background dominance muammosini yetarli hal qilmagan.

---

# 31. V2 yaxshilanishlar

```text
256 → 512 resolution
kamroq downsampling
Weighted BCE
Tversky Loss
batch = 4
learning rate = 0.0002
```

Model keyin asta-sekin yaxshilangan.

Misol validation Dice:

```text
Epoch 1  ≈ 0.0027
Epoch 12 ≈ 0.0511
Epoch 15 ≈ 0.1419
Epoch 18 ≈ 0.1847
Epoch 21 ≈ 0.1960
Epoch 22 ≈ 0.2044
```

Bu qiymatlar training vaqtida config threshold 0.5 bilan kuzatilgan.

---

# 32. `train.py` resume buglari va tuzatish

Dastlab resume paytida `history=[]` sabab oldingi history overwrite bo‘lgan. Keyinchalik oldingi CSV yuklanib davom etadigan qilindi.

Yana dastlab `last_checkpoint` best/stale state yangilanishidan oldin save bo‘lgan. Final tartib:

```text
validation
 ↓
best_val / stale update
 ↓
best model save
 ↓
last checkpoint save
```

qilib tuzatildi.

Final `training_history.csv` 1–25 epochni to‘liq saqlaydi.

---

# 33. Threshold tuning

Default `0.5` optimal bo‘lmagan.

Validation searchda:

```text
0.50 → global Dice ≈ 0.197
0.80 → ≈ 0.502
0.90 → ≈ 0.587
0.95 → ≈ 0.625
```

Fine search natijasida:

```text
threshold = 0.993
```

tanlangan.

Best model validation global Dice threshold tuningda taxminan `0.6483` bo‘lgan.

---

# 34. Connected component analysis

Validation ground-truth masklarda:

```text
Jami component       : 13925
Median / image       : 100.5
Mean / image         : 110.5
Median component size: 1 pixel
```

Tiny componentlar:

```text
< 2 px  : 69.59%
< 3 px  : 85.80%
< 5 px  : 95.20%
< 10 px : 98.15%
```

Shu sabab `stone count` metric ishonchli emasligi aniqlandi.

---

# 35. Nega `mean_count_abs_error` olib tashlandi?

Oldingi evaluationda `mean_count_abs_error ≈ 102` chiqgan. Bu model 102 ta stone xato qildi degani emas. Juda ko‘p 1-pixellik annotation/noise component alohida obyekt sifatida sanalgan.

Final projectda klinik “stone count” berilmaydi. Faqat:

```text
Estimated connected regions
```

deb ko‘rsatiladi.

---

# 36. Postprocessing tuning

Validationda `min_component_pixels` qidirilgan.

Muhim natijalar:

```text
0 px:
Dice      = 0.6483
Precision = 0.5992
Recall    = 0.7063

50 px:
Dice      = 0.6520
Precision = 0.6122
Recall    = 0.6974
Zero predictions = 0

75 px:
Dice      = 0.6423
Recall    = 0.6692
Zero predictions = 5
```

Final:

```text
min_component_pixels = 50
```

---

# 37. Final validation

```text
Global Dice      = 0.6520
Global IoU       = 0.4837
Global Precision = 0.6122
Global Recall    = 0.6974
```

---

# 38. `src/evaluate.py`

Final test evaluation script.

Hisoblaydi:

### Mean per-image

```text
Mean Dice
Mean IoU
Mean Precision
Mean Recall
```

### Global

```text
Global Dice
Global IoU
Global Precision
Global Recall
```

### Quantitative

```text
Mean absolute area error in pixels
```

Run:

```bash
python src/evaluate.py --config configs/config.yaml
```

---

# 39. Final test natijalari

Test images:

```text
126
```

Mean per-image:

```text
Dice      = 0.6027
IoU       = 0.4478
Precision = 0.5596
Recall    = 0.6887
```

Global:

```text
Dice      = 0.6636
IoU       = 0.4966
Precision = 0.6174
Recall    = 0.7173
```

Area:

```text
Mean absolute area error = 90.94 pixels
```

Global va mean metric bir xil emas: global barcha pixelni bitta katta confusion hisobida ko‘radi, mean esa har imagega teng vazn beradi.

---

# 40. Error analysis

Worst case:

```text
880.tif
Dice      = 0.1579
Precision = 0.2542
Recall    = 0.1145
```

Median misol:

```text
375.tif
Dice      = 0.6061
Precision = 0.5466
Recall    = 0.6801
```

Best case:

```text
312.tif
Dice      = 0.8616
Precision = 0.8431
Recall    = 0.8809
```

Bu case’lar `best_median_worst_cases.png`da vizual ko‘rsatilgan.

---

# 41. `src/predict.py`

Yangi TIF image uchun inference.

```text
TIF
 ↓
preprocess
 ↓
model
 ↓
sigmoid
 ↓
threshold 0.993
 ↓
remove <50 px
 ↓
final mask
 ↓
area + connected regions
 ↓
save outputs
```

Run:

```bash
python src/predict.py \
  --config configs/config.yaml \
  --image path/to/image.tif
```

Misol `312.tif`:

```text
Predicted area = 816 pixels
Estimated connected regions = 1
```

---

# 42. `src/inference_utils.py`

Inferencega oid reusable yordamchi logic uchun modul. Final inference oqimining asosiy boshqaruvi `predict.py` va `app.py`da bajariladi.

---

# 43. `app/app.py` — Gradio

Final web UI.

Foydalanuvchi TIF upload qiladi va quyidagilarni ko‘radi:

- Preprocessed CT;
- Probability map;
- Final segmentation mask;
- Stone overlay;
- Predicted stone area;
- Estimated connected regions;
- Region areas;
- threshold va postprocessing qiymati.

`312.tif` bilan CLI va Gradio bir xil natija bergan:

```text
816 pixels
1 connected region
```

Bu inference consistency’ni tasdiqlaydi.

---

# 44. `run_pipeline.py`

Project bosqichlarini ketma-ket ishga tushirish uchun top-level entry point. Konseptual flow:

```text
download → prepare → inspect → train → evaluate
```

Amaliy development davomida har bosqich alohida run qilingan, chunki har qadam natijasi tekshirilgan.

---

# 45. `requirements.txt`

Final dependencylar:

```text
monai>=1.4,<2.0
numpy>=1.26,<3.0
pandas>=2.1,<3.0
scikit-learn>=1.4,<2.0
scipy>=1.12,<2.0
scikit-image>=0.22,<1.0
matplotlib>=3.8,<4.0
pyyaml>=6.0,<7.0
tqdm>=4.66,<5.0
pillow>=10.0,<12.0
gradio>=4.44,<7.0
kagglehub>=0.3.6,<1.0
```

PyTorch Colabda CUDA bilan oldindan mavjud bo‘lgani uchun requirements ichida pin qilinmagan.

---

# 46. `.gitignore`

GitHub’ga yuborilmaydi:

```text
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
venv/
.env
.gradio/
*.pem
raw dataset
machine-specific split/manifest
last checkpoint
prediction temp outputlar
```

Final best model uchun exception bor:

```text
!models/best_2d_unet.pth
```

Shu sabab final model GitHub repositoryda saqlanishi mumkin.

---

# 47. Virtual environment

Colabda alohida virtual environment ishlatilmagan. Colab runtime o‘zi izolyatsiyalangan environment sifatida ishlatilgan.

VS Code’da `.venv` tavsiya qilinadi:

```bash
python -m venv .venv
```

Git Bash:

```bash
source .venv/Scripts/activate
```

Keyin:

```bash
pip install -r requirements.txt
```

`.env` boshqa narsa: secret/environment variable uchun. Bu projectda `.env` majburiy emas.

---

# 48. 0 dan oxirigacha bajarilgan ishlar

1. Kidney stone segmentation muammosi tanlandi.
2. Dastlab 3D dataset ko‘rib chiqildi.
3. Restricted access sabab KSSD2025 tanlandi.
4. Modular project strukturasi yaratildi.
5. Dataset yuklandi.
6. Dataset inspect qilindi.
7. 838 image-mask pair topildi.
8. `manifest.csv` yaratildi.
9. 586/126/126 split qilindi.
10. Sample overlay bilan pairing tekshirildi.
11. Initial 256 × 256 U-Net train qilindi.
12. Dice ≈ 0.02 bo‘lib, muammo analiz qilindi.
13. Foreground ratio ≈ 0.135% ekani topildi.
14. Input 512 × 512 qilindi.
15. Downsampling kamaytirildi.
16. Weighted BCE + Tversky lossga o‘tildi.
17. 25 epoch training qilindi.
18. Best checkpoint saqlandi.
19. Validationda threshold tuning qilindi.
20. Final threshold `0.993` tanlandi.
21. Ground-truth component analysis qilindi.
22. Tiny componentlar juda ko‘p ekani aniqlandi.
23. Stone-count metric olib tashlandi.
24. Validationda postprocessing tuning qilindi.
25. `min_component_pixels=50` tanlandi.
26. Final test evaluation qilindi.
27. Mean va global metriclar chiqarildi.
28. Error analysis qilindi.
29. Best/median/worst visualization yaratildi.
30. `predict.py` final qilindi.
31. Gradio app yaratildi.
32. `312.tif` bilan CLI/Gradio consistency tekshirildi.
33. `evaluate.py`dan noto‘g‘ri count metric olib tashlandi.
34. `train.py` resume buglari tuzatildi.
35. Training history 1–25 epoch tiklandi.
36. README va structure hujjatlari final qilindi.
37. `.gitignore` GitHub uchun tozalandi.
38. Full va portfolio ZIP yaratildi.
39. Portfolio GitHub repositoryga push qilindi.
40. Ushbu batafsil `projectga_izoh.md` tayyorlandi.

---

# 49. Google Colab’da run qilish

Drive mount:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Projectga o‘tish:

```python
%cd /content/drive/MyDrive/kidney_stone_project_2
```

Dependency:

```python
!pip install -q -r requirements.txt
```

GPU:

```python
!nvidia-smi
```

Environment:

```python
!python src/check_environment.py
```

Dataset:

```python
!python src/download_data.py --config configs/config.yaml
!python src/prepare_data.py --config configs/config.yaml
!python src/visualize_sample.py --config configs/config.yaml
```

Training:

```python
!python src/train.py --config configs/config.yaml
```

Resume:

```python
!python src/train.py --config configs/config.yaml --resume
```

Evaluation:

```python
!python src/evaluate.py --config configs/config.yaml
```

Prediction:

```python
!python src/predict.py \
    --config configs/config.yaml \
    --image data/raw/kssd2025/data/image/312.tif
```

Gradio:

```python
!python app/app.py
```

---

# 50. VS Code’da run qilish

Project papkasini VS Code’da oching.

Tavsiya etilgan environment:

```bash
python -m venv .venv
```

Git Bash:

```bash
source .venv/Scripts/activate
```

Pip:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch localda bo‘lmasa CPU yoki GPU konfiguratsiyaga mos alohida o‘rnatiladi.

Environment check:

```bash
python src/check_environment.py
```

Gradio:

```bash
python app/app.py
```

Local browser URL odatda:

```text
http://127.0.0.1:7860
```

Portfolio versiyada raw dataset yo‘q. Training uchun datasetni qayta download qilish kerak. Inference uchun esa alohida `.tif` image upload qilish mumkin.

---

# 51. GitHub workflow

Yangi o‘zgarish qo‘shilganda:

```bash
git status
git add .
git commit -m "Update project"
git push
```

Masalan ushbu faylni qo‘shish:

```bash
git add projectga_izoh.md
git commit -m "Add detailed Uzbek project documentation"
git push
```

---

# 52. Final model konfiguratsiyasi

```text
Task                : 2D semantic segmentation
Model               : 2D U-Net
Input               : 512 × 512 × 1
Output              : 512 × 512 × 1
Channels            : 16 → 32 → 64 → 128
Loss                : Weighted BCE + Tversky
Epochs              : 25
Batch size          : 4
Learning rate       : 0.0002
Weight decay        : 0.00001
AMP                 : True
Best model          : models/best_2d_unet.pth
Threshold           : 0.993
Min component       : 50 pixels
```

---

# 53. Final natijalar

## Validation

```text
Global Dice      = 0.6520
Global IoU       = 0.4837
Global Precision = 0.6122
Global Recall    = 0.6974
```

## Test — Mean per image

```text
Dice      = 0.6027
IoU       = 0.4478
Precision = 0.5596
Recall    = 0.6887
```

## Test — Global

```text
Dice      = 0.6636
IoU       = 0.4966
Precision = 0.6174
Recall    = 0.7173
```

## Area

```text
Mean absolute area error = 90.94 pixels
```

---

# 54. Projectning kuchli tomonlari

- real medical image segmentation task;
- modular project structure;
- train/val/test separation;
- class imbalance analysis;
- baseline failure analysis;
- model/loss/resolution improvement;
- validation-only threshold tuning;
- validation-only postprocessing tuning;
- test leakagega ehtiyotkor yondashuv;
- mean va global metriclar;
- error analysis;
- prediction script;
- Gradio web UI;
- GitHub-ready portfolio structure;
- batafsil documentation.

---

# 55. Muhim limitationlar

1. **2D model** — 3D context yo‘q.
2. **Physical spacing yo‘q** — area faqat pixelda.
3. **Stone-negative cases yetishmaydi** — false-positive behavior to‘liq baholanmagan.
4. **Patient ID yo‘q** — patient-level split qilinmagan.
5. **Probability calibrated emas** — 0.993 confidence emas.
6. **Connected regions stone count emas**.
7. **Clinical deployment emas** — research, learning va portfolio uchun.

---

# 56. Kelajakdagi yaxshilanishlar

- patient-level split;
- stone-negative CT case’lar;
- larger dataset;
- cross-validation;
- Attention U-Net;
- U-Net++;
- nnU-Net;
- SegResNet;
- pretrained encoder;
- probability calibration;
- ensemble;
- uncertainty estimation;
- external validation;
- CT physical spacing;
- 3D U-Net;
- mm² area;
- mm³ stone volume;
- clinical expert review.

---

# 57. Yakuniy xulosa

`kidney_stone_project_2` oddiy notebook tajribasi emas. U to‘liq end-to-end Computer Vision / Medical Image Segmentation pipeline hisoblanadi:

```text
Problem definition
      ↓
Dataset selection
      ↓
Data inspection
      ↓
Data preparation
      ↓
Train / Val / Test
      ↓
Preprocessing
      ↓
Baseline
      ↓
Failure analysis
      ↓
Improved U-Net + imbalance-aware loss
      ↓
Training
      ↓
Validation tuning
      ↓
Threshold = 0.993
      ↓
Postprocessing = 50 pixels
      ↓
Final test
      ↓
Error analysis
      ↓
Prediction
      ↓
Gradio
      ↓
Documentation
      ↓
GitHub portfolio
```

Final asosiy natija:

```text
Global Test Dice = 0.6636
Mean Test Dice   = 0.6027
```

Final deployment modeli:

```text
models/best_2d_unet.pth
```

Final user flow:

```text
CT TIF image
   ↓
Gradio / predict.py
   ↓
2D U-Net
   ↓
Stone segmentation mask
   ↓
Pixel area
   ↓
Overlay
   ↓
Estimated connected regions
```

Shu bilan loyiha datasetdan boshlab model training, validation, test, inference, visualization, web UI, documentation va GitHubgacha to‘liq yakunlangan.
