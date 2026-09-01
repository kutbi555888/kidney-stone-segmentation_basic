# Kidney Stone Segmentation — Project Structure

## 1. Hujjat maqsadi

Ushbu fayl `kidney_stone_project_2` loyihasining ichki tuzilishini tushuntiradi.

Bu yerda quyidagilar yozilgan:

- har bir papka nima uchun kerak;
- har bir asosiy Python fayli nima qiladi;
- dataset qayerda joylashadi;
- train / validation / test qanday ishlaydi;
- model qayerga saqlanadi;
- evaluation natijalari qayerda saqlanadi;
- prediction va Gradio qanday ishlaydi;
- loyiha 0 dan final holatgacha qanday qurilgan.

`README.md` loyiha haqida umumiy hujjat bo‘lsa, ushbu fayl ko‘proq:

> **project architecture + file responsibilities + workflow**

uchun mo‘ljallangan.

---

# 2. Asosiy project structure

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
│   │
│   ├── processed/
│   │   └── manifest.csv
│   │
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
├── PROJECT_STRUCTURE.md
├── README.md
├── requirements.txt
└── run_pipeline.py
3. app/
app/
└── app.py

Bu papka foydalanuvchi interfeysi uchun ishlatiladi.

app/app.py

Final Gradio web application.

Asosiy vazifalari:

foydalanuvchidan CT TIF image qabul qilish;
final config faylni o‘qish;
best_2d_unet.pth modelni yuklash;
training bilan bir xil preprocessing qo‘llash;
model inference bajarish;
sigmoid probability olish;
final threshold qo‘llash;
kichik connected componentlarni olib tashlash;
final segmentation mask yaratish;
stone pixel area hisoblash;
estimated connected regionlarni hisoblash;
overlay yaratish;
natijani Gradio UI orqali ko‘rsatish.

Final inference parametrlari:

Threshold:
0.993

Minimum connected component:
50 pixels

App modelni har bir prediction vaqtida qayta yuklamaydi.

Model:

app start
    ↓
model load
    ↓
memory'da saqlanadi
    ↓
har requestda inference

Bu Gradio ishlashini tezlashtiradi.

4. configs/
configs/
└── config.yaml

Loyihaning asosiy konfiguratsiyasi shu papkada saqlanadi.

configs/config.yaml

Bu loyiha uchun central configuration fayl.

Asosiy bo‘limlari:

project:
data:
training:
model:
postprocess:
paths:
project

Project nomi va random seed.

Misol:

project:
  name: kidney_stone_2d_segmentation_kssd2025
  seed: 42
data

Dataset va split parametrlarini saqlaydi.

Misol:

data:
  raw_dir: data/raw/kssd2025
  manifest_csv: data/processed/manifest.csv
  splits_dir: data/splits
  test_size: 0.15
  val_size: 0.15
  image_size: [512, 512]

Final model uchun:

Input image size:
512 × 512
training

Training parametrlarini saqlaydi.

Final asosiy qiymatlar:

epochs        = 25
batch_size    = 4
learning_rate = 0.0002
weight_decay  = 0.00001
AMP           = True
threshold     = 0.993
model

U-Net model parametrlarini saqlaydi.

Final konfiguratsiya:

Input channels  : 1
Output channels : 1

Channels:
16 → 32 → 64 → 128

Downsampling:
3
postprocess

Predictiondan keyingi filtering.

postprocess:
  min_component_pixels: 50

Bu qiymat validation set orqali tanlangan.

paths

Model va result pathlarini saqlaydi.

Misol:

models/best_2d_unet.pth
models/last_checkpoint.pth

results/metrics/test_metrics.csv
results/metrics/test_summary.json

results/predictions/
5. data/
data/
├── raw/
├── processed/
└── splits/

Dataset bilan bog‘liq barcha fayllar shu papkada saqlanadi.

6. data/raw/
data/raw/
└── kssd2025/

Original dataset shu yerda saqlanadi.

Final dataset strukturasi:

data/raw/kssd2025/data/
├── image/
└── label/
image/

Original CT TIF tasvirlar.

Misol:

1.tif
10.tif
306.tif
312.tif
1321.tif
...

Bu fayllar model input hisoblanadi.

label/

Ground-truth stone segmentation masklar.

Misol:

1.tif
10.tif
306.tif
312.tif
1321.tif
...

Muhim:

image/312.tif

CT image.

label/312.tif

Ground-truth segmentation mask.

Ular bir xil filename orqali pair qilinadi.

7. data/processed/
data/processed/
└── manifest.csv

Dataset tekshirilgandan keyin yaratiladigan umumiy metadata.

manifest.csv

Har image uchun uning mos mask pathini saqlaydi.

Misol:

image
mask
stone_pixels

Taxminiy row:

image/.../312.tif
label/.../312.tif
...

Bu manifest dataset preparation bosqichining natijasi.

8. data/splits/
data/splits/
├── train.csv
├── val.csv
└── test.csv

Dataset 3 qismga ajratilgan.

Final split:

Train      : 586
Validation : 126
Test       : 126
Total      : 838
train.csv

Faqat model training uchun ishlatiladi.

val.csv

Quyidagilar uchun ishlatiladi:

model checkpoint tanlash;
threshold tuning;
postprocessing tuning;
training monitoring.

Test set bu parametrlarni tanlash uchun ishlatilmaydi.

test.csv

Faqat final evaluation uchun ishlatiladi.

Final model:

best_2d_unet.pth
+
threshold = 0.993
+
min_component_pixels = 50

freeze qilingandan keyin test set baholangan.

9. Split limitation

Dataset filename'larida patient ID mavjud emas.

Shuning uchun split:

image-level

amalga oshirilgan.

Agar bitta patientdan bir nechta CT slice mavjud bo‘lsa,
ular turli splitlarga tushgan bo‘lishi ehtimolini to‘liq
inkor qilib bo‘lmaydi.

Kelajakdagi versiyada patient-level metadata bo‘lsa:

patient-level split

ishlatish kerak.

10. models/
models/
├── best_2d_unet.pth
└── last_checkpoint.pth

Training vaqtida yaratiladigan model checkpointlar.

best_2d_unet.pth

Final model.

Validation Dice bo‘yicha eng yaxshi checkpoint.

Inference, evaluation va Gradio aynan shu modeldan foydalanadi.

Final deployment model:

models/best_2d_unet.pth
last_checkpoint.pth

Trainingning eng oxirgi epoch checkpointi.

Bu fayl asosan:

training resume

uchun kerak.

Best model bilan bir xil bo‘lishi shart emas.

Masalan:

best model:
epoch ~22

last checkpoint:
epoch 25

bo‘lishi mumkin.

Final inference uchun:

best_2d_unet.pth

ishlatiladi.

11. results/
results/
├── figures/
├── metrics/
└── predictions/

Trainingdan keyingi barcha natijalar shu yerda saqlanadi.

12. results/figures/

Visualization natijalari.

Masalan:

sample_overlay.png
best_median_worst_cases.png
sample_overlay.png

Datasetdagi CT image va ground-truth mask alignmentni
tekshirish uchun ishlatilgan visualization.

Maqsad:

image-mask pairing

to‘g‘ri ekanini ko‘z bilan tekshirish.

best_median_worst_cases.png

Final test dataset error analysis visualization.

Unda:

Worst cases
Median cases
Best cases

ko‘rsatiladi.

Har case uchun:

Original CT
Ground Truth
Prediction
GT + Prediction overlay

mavjud.

Bu faqat metric raqamlariga emas,
modelning real segmentation xatti-harakatiga qarash imkonini beradi.

13. results/metrics/
results/metrics/
├── training_history.csv
├── test_metrics.csv
└── test_summary.json
training_history.csv

Epoch bo‘yicha training natijalari.

Odatda:

epoch
train_loss
val_loss
val_dice

kabi qiymatlarni saqlaydi.

test_metrics.csv

Har bir test image uchun alohida metrikalar.

Final ustunlar:

image
mask
dice
iou
precision
recall
true_area_pixels
predicted_area_pixels
area_abs_error_pixels

Masalan:

312.tif

Dice:
0.8616

Precision:
0.8431

Recall:
0.8809
test_summary.json

Final project natijalari.

Final test natijalari:

n_test:
126

mean_dice:
0.6027

mean_iou:
0.4478

mean_precision:
0.5596

mean_recall:
0.6887

global_dice:
0.6636

global_iou:
0.4966

global_precision:
0.6174

global_recall:
0.7173

mean_area_abs_error_pixels:
90.94

Shuningdek final pipeline parametrlari ham saqlanadi:

threshold:
0.993

min_component_pixels:
50
14. results/predictions/

Yangi CT image predictionlari.

Masalan:

312_mask.png
312_probability.png
312_prediction.png
*_mask.png

Final binary segmentation mask.

Pipeline:

probability
    ↓
threshold
    ↓
small component filtering
    ↓
final mask
*_probability.png

Modelning sigmoid output probability map.

Muhim:

Bu qiymatlarni:

clinical confidence

deb talqin qilish mumkin emas.

Probability output kalibrlanmagan.

*_prediction.png

Bir nechta natijani bir figure ichida ko‘rsatadi:

CT
Probability map
Final mask
Overlay
15. src/
src/

Loyihaning asosiy Python kodlari shu papkada.

16. src/check_environment.py

Environmentni tekshiradi.

Masalan:

Python;
PyTorch;
CUDA;
GPU mavjudligi;
kerakli package'lar.

Colab environmentni tekshirish uchun ishlatiladi.

17. src/common.py

Project bo‘ylab qayta ishlatiladigan umumiy utility funksiyalar.

Asosiy vazifalar:

config o‘qish;
project root aniqlash;
relative pathni absolute pathga aylantirish;
reproducibility uchun seed;
kerakli papkalarni yaratish.

Misol:

load_config(...)
resolve_path(...)

kabi funksiyalar boshqa scriptlarda ishlatiladi.

18. src/data_utils.py

Dataset va data preparation bilan bog‘liq yordamchi funksiyalar.

Projectning turli bosqichlarida:

file matching
path processing
metadata
dataset helpers

uchun ishlatilishi mumkin.

19. src/dataset.py

PyTorch Dataset class.

Bu projectdagi eng muhim data fayllardan biri.

Asosiy vazifalari:

CT image o‘qish;
corresponding mask o‘qish;
grayscale qilish;
percentile clipping;
normalization;
resize;
binary mask yaratish;
augmentation;
PyTorch tensor qaytarish.

Final preprocessing:

TIF image
   ↓
grayscale
   ↓
float32
   ↓
0.5 / 99.5 percentile clipping
   ↓
0–1 normalization
   ↓
512 × 512 resize
   ↓
Tensor

Mask uchun:

mask > 0
   ↓
binary
   ↓
nearest-neighbor resize

Training augmentation:

horizontal flip
20. Nega 512 × 512?

Dastlab kichik image size bilan model stone regionlarni yaxshi
o‘rganmadi.

Kidney stone tasvirning juda kichik qismini egallaydi.

Shuning uchun final pipeline:

512 × 512

ishlatadi.

Bu kichik foreground regionlarni saqlashga yordam beradi.

21. src/download_data.py

KSSD2025 datasetni yuklab olish uchun ishlatiladi.

Dataset mavjud bo‘lmasa:

download
    ↓
extract
    ↓
raw directory

pipeline ishlaydi.

22. src/inspect_data.py

Datasetni dastlabki tekshirish uchun.

Tekshiriladigan narsalar:

image soni;
mask soni;
image-mask matching;
image shape;
mask values;
foreground mavjudligi;
dataset structure.
23. src/prepare_data.py

Dataset preparationning asosiy scripti.

Asosiy bosqichlari:

Raw dataset
    ↓
image file qidirish
    ↓
matching label topish
    ↓
stone pixel count
    ↓
manifest.csv
    ↓
train.csv
    ↓
val.csv
    ↓
test.csv

Final natija:

838 image-mask pair

Split:

586 train
126 validation
126 test
24. src/visualize_sample.py

Datasetdan sample image olib visualization qiladi.

Maqsad:

CT image;
mask;
overlay;

bir-biriga to‘g‘ri mos kelayotganini tekshirish.

Bu bosqich trainingdan oldin juda muhim.

25. src/model.py

Final segmentation modelini yaratadi.

Model:

2D U-Net

U-Netning asosiy g‘oyasi:

Encoder
   ↓
feature extraction
   ↓
bottleneck
   ↓
Decoder
   ↓
pixel-level segmentation

Skip connections encoder featurelarini decoderga uzatadi.

Bu boundary va spatial detailni tiklashga yordam beradi.

Final channels:

16
32
64
128

Juda chuqur downsampling ishlatilmaydi.

Sababi:

stone regionlari juda kichik

bo‘lishi mumkin.

26. src/losses.py

Final training loss.

Dastlab oddiy BCE + Dice ishlatilgan.

Lekin stone foreground ratio juda kichik:

~0.135%

bo‘lgani sabab model backgroundni o‘rganishga moyil bo‘lgan.

Final loss:

Weighted BCE
+
Tversky Loss

Kombinatsiya:

0.4 × Weighted BCE
+
0.6 × Tversky

Weighted BCE:

pos_weight = 50

Tversky:

alpha = 0.3
beta  = 0.7

Bu kichik foreground va false-negative muammosiga qarshi ishlatilgan.

27. src/metrics.py

Segmentation metric helper funksiyalari.

Masalan:

Dice;
IoU;
Precision;
Recall;
connected component processing.

Final evaluation uchun asosiy hisoblashlar evaluate.py
ichida ham to‘liq va aniq qayta yozilgan.

28. src/train.py

Model training pipeline.

Asosiy flow:

config
   ↓
train / val CSV
   ↓
Dataset
   ↓
DataLoader
   ↓
U-Net
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

Final training:

Epochs:
25

Batch:
4

Learning rate:
0.0002

AMP:
True

Google Colab GPU uchun moslashtirilgan.

29. Training davomida kuzatilgan muammo

Dastlab model:

Dice ≈ 0.02

atrofida qolgan.

Loss pasaygan bo‘lsa ham segmentation quality yaxshi bo‘lmagan.

Sabablar:

foreground juda kichik;
class imbalance;
image resolution yetarli emas;
kichik stone featurelari downsamplingda yo‘qolishi mumkin.

Final yaxshilanishlar:

256 → 512 resolution
deeper downsampling kamaytirildi
Weighted BCE
Tversky
threshold tuning
postprocessing tuning

natijada model ancha yaxshilandi.

30. src/evaluate.py

Final test evaluation script.

Bu script endi quyidagilarni hisoblaydi:

Per-image mean metrics
Mean Dice
Mean IoU
Mean Precision
Mean Recall
Global metrics
Global Dice
Global IoU
Global Precision
Global Recall
Quantitative metric
Mean absolute area error in pixels

Final test:

Mean Dice       = 0.6027
Mean IoU        = 0.4478
Mean Precision  = 0.5596
Mean Recall     = 0.6887

Global Dice      = 0.6636
Global IoU       = 0.4966
Global Precision = 0.6174
Global Recall    = 0.7173

Area MAE:
90.94 pixels
31. Nega global va mean Dice farq qiladi?

Mean Dice:

har bir image Dice
        ↓
o'rtacha

Har bir image bir xil vaznga ega.

Global Dice:

barcha test pixellar
        ↓
bitta katta confusion hisob

Katta masklar global metricda ko‘proq ta’sir qiladi.

Shuning uchun:

Mean Dice:
0.6027

Global Dice:
0.6636

bir xil chiqmasligi normal.

32. src/predict.py

Yangi image uchun final inference.

Pipeline:

TIF image
    ↓
preprocess
    ↓
U-Net
    ↓
sigmoid
    ↓
threshold = 0.993
    ↓
remove components < 50 px
    ↓
final mask

Keyin hisoblanadi:

Predicted area pixels
Estimated connected regions
Region areas

Va saqlanadi:

mask
probability map
prediction visualization
33. Final 312.tif test

Inference consistency test uchun:

312.tif

ishlatilgan.

Natija:

Predicted stone area:
816 pixels

Estimated connected regions:
1

CLI predict.py va Gradio ikkalasi ham:

816 pixels
1 connected region

natija berdi.

Bu inference pipeline bir xil ekanini tasdiqlaydi.

34. Connected region limitation

Datasetdagi ground-truth masklarda juda ko‘p kichik connected
componentlar mavjud.

Validation analysis:

Total components:
13925

Median component size:
1 pixel

Taxminan:

< 2 pixels:
69.59%

< 3 pixels:
85.80%

< 5 pixels:
95.20%

Shuning uchun connected-component countni:

real kidney stone count

deb talqin qilish mumkin emas.

Final projectda:

Estimated connected regions

deb ko‘rsatiladi.

35. Threshold tuning

Default threshold:

0.5

model uchun yaxshi natija bermagan.

Validation search natijasida probability threshold oshirilgan.

Final:

threshold = 0.993

Validation global Dice:

~0.648

atrofida bo‘lgan.

Muhim:

0.993 ≠ 99.3% clinical confidence

Model probabilitylari kalibrlanmagan.

36. Postprocessing tuning

Validation datasetda turli:

min_component_pixels

qiymatlari tekshirilgan.

Misollar:

0
5
10
20
30
40
50
60
75
100
...

Final tanlov:

50 pixels

Validation natija:

Dice:
0.6520

IoU:
0.4837

Precision:
0.6122

Recall:
0.6974

Zero predictions:
0

75 pixel va undan yuqori qiymatlarda kichik haqiqiy regionlarni
yo‘qotish kuchaygan.

Shu sabab:

50

final balans sifatida tanlangan.

37. src/inference_utils.py

Inference bilan bog‘liq yordamchi funksiyalar uchun ajratilgan fayl.

Project versiyasiga qarab:

preprocessing helpers;
mask processing;
prediction utilities;

kabi kodlarni saqlashi mumkin.

Final predict.py kerakli preprocessing va postprocessingni
o‘z ichida aniq boshqaradi.

38. run_pipeline.py

Bir nechta project bosqichlarini ketma-ket ishga tushirish uchun
entry-point sifatida ishlatiladi.

Masalan konseptual flow:

download
    ↓
prepare
    ↓
inspect
    ↓
train
    ↓
evaluate

Final Colab workflowda har bir bosqich alohida run qilingan,
chunki har qadam natijasi tekshirilgan.

39. requirements.txt

Project dependencylari.

Masalan:

numpy;
pandas;
scipy;
Pillow;
matplotlib;
tqdm;
PyYAML;
Gradio;
MONAI yoki project model dependencylari.

Google Colabda CUDA-compatible PyTorch allaqachon mavjud
bo‘lishi mumkin.

Shu sabab PyTorchni qayta o‘rnatishda ehtiyot bo‘lish kerak.

40. .gitignore

Git repositoryga kiritilmasligi kerak bo‘lgan fayllarni
belgilaydi.

Masalan:

__pycache__/
.ipynb_checkpoints/
large dataset
temporary files
local environment

Katta raw dataset va model fayllarini GitHubga qo‘yishda
repository size limitlariga e’tibor berish kerak.

41. notebooks/
notebooks/
└── COLAB_RUN_KSSD2025.ipynb

Notebook projectni Google Colab muhitida ishlatishga yordam beradi.

Lekin asosiy production kod:

src/

va:

app/

ichida saqlanadi.

Notebook project logicning yagona manbasi emas.

42. Project development workflow

Loyiha 0 dan quyidagi ketma-ketlikda qurildi.

Bosqich 1 — Dataset tanlash

Dastlab 3D kidney-stone CT dataset ko‘rib chiqilgan.

Lekin dataset restricted access bo‘lgani sabab
ochiq KSSD2025 datasetga o‘tilgan.

Natijada task:

3D segmentation

dan:

2D semantic segmentation

ga moslashtirilgan.

Bosqich 2 — Dataset download

KSSD2025 raw fayllari yuklab olingan.

Bosqich 3 — Dataset inspection

Image va mask fayllari tekshirilgan.

Aniqlangan:

838 image
838 corresponding mask
Bosqich 4 — Pairing

Filename orqali:

image
↔
label

pairing yaratilgan.

Bosqich 5 — Manifest

manifest.csv yaratilgan.

Bosqich 6 — Split

Dataset:

Train      586
Validation 126
Test       126

ga ajratilgan.

Bosqich 7 — Visual inspection

CT + mask overlay ko‘rilgan.

Pairing va alignment tekshirilgan.

Bosqich 8 — Initial model

Dastlabki U-Net:

256 × 256

va oddiyroq loss bilan train qilingan.

Natija juda past:

Dice ≈ 0.02
Bosqich 9 — Foreground analysis

Stone foreground juda kichik ekani aniqlangan:

~0.135%

Bu kuchli class imbalance ekanini ko‘rsatgan.

Bosqich 10 — V2 model

Yaxshilanishlar:

512 × 512
3 downsampling
Weighted BCE
Tversky Loss
conservative augmentation
batch = 4
Bosqich 11 — Training

25 epochgacha training davom ettirilgan.

Best model validation Dice asosida saqlangan.

Bosqich 12 — Threshold tuning

Validationda threshold qidirilgan.

Final:

0.993
Bosqich 13 — Component analysis

Ground-truth masklarda juda ko‘p tiny component aniqlangan.

Bu sababli direct stone-count metric bekor qilingan.

Bosqich 14 — Postprocessing tuning

Validationda:

min_component_pixels

tanlangan.

Final:

50
Bosqich 15 — Final test

Model va parametrlar freeze qilingan.

Keyin test set bir marta final baholangan.

Final:

Global Dice:
0.6636

Mean Dice:
0.6027
Bosqich 16 — Error analysis

Worst / median / best case'lar ajratilgan.

Misollar:

Worst:
880.tif

Median:
375.tif

Best:
312.tif
Bosqich 17 — Inference

predict.py final qilingan.

Bosqich 18 — Gradio

Web application yaratilgan.

312.tif bilan CLI va Gradio natijasi solishtirilgan.

Natija mos:

Area:
816 pixels

Estimated regions:
1
43. Final end-to-end architecture
                    KSSD2025
                        │
                        ▼
                 Raw CT + Label
                        │
                        ▼
                prepare_data.py
                        │
                        ▼
                  manifest.csv
                        │
                        ▼
              Train / Val / Test
                        │
                        ▼
                  dataset.py
                        │
           ┌────────────┴────────────┐
           │                         │
           ▼                         ▼
      Preprocessing              Augmentation
           │
           └────────────┬────────────┘
                        ▼
                    2D U-Net
                        │
                        ▼
          Weighted BCE + Tversky
                        │
                        ▼
                     Train
                        │
                        ▼
              best_2d_unet.pth
                        │
                        ▼
               Validation tuning
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
      threshold = 0.993     min_component = 50
             │                     │
             └──────────┬──────────┘
                        ▼
                  Final pipeline
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
          evaluate.py          predict.py
              │                   │
              ▼                   ▼
         Test metrics         Final mask
                                  │
                                  ▼
                              Gradio UI
44. Final inference architecture
User TIF CT image
        │
        ▼
Percentile clipping
        │
        ▼
Normalization
        │
        ▼
Resize 512 × 512
        │
        ▼
2D U-Net
        │
        ▼
Logits
        │
        ▼
Sigmoid
        │
        ▼
Probability map
        │
        ▼
Threshold = 0.993
        │
        ▼
Binary mask
        │
        ▼
Remove component < 50 pixels
        │
        ▼
Final stone segmentation
        │
        ├───────────────┐
        ▼               ▼
Pixel area          Overlay
        │
        ▼
Estimated connected regions
45. Final project metrics
Validation
Global Dice:
0.6520

Global IoU:
0.4837

Global Precision:
0.6122

Global Recall:
0.6974
Test — Mean per-image
Dice:
0.6027

IoU:
0.4478

Precision:
0.5596

Recall:
0.6887
Test — Global
Dice:
0.6636

IoU:
0.4966

Precision:
0.6174

Recall:
0.7173
Area
Mean absolute error:
90.94 pixels
46. Final model configuration
Task:
2D semantic segmentation

Model:
2D U-Net

Input:
512 × 512 × 1

Output:
512 × 512 × 1

Loss:
Weighted BCE + Tversky

Epochs:
25

Batch:
4

Learning rate:
0.0002

Best checkpoint:
models/best_2d_unet.pth

Threshold:
0.993

Postprocessing:
remove connected components < 50 pixels
47. Muhim limitationlar
47.1 2D model

Model faqat bitta axial slice bilan ishlaydi.

3D volumetric context ishlatilmaydi.

47.2 Physical spacing mavjud emas

Natija:

pixel area

birligida.

Quyidagilar hisoblanmaydi:

mm²
mm³
47.3 Negative cases

Datasetdagi foydalanilgan masklarda foreground mavjud.

Shuning uchun stone bo‘lmagan CT image'larda false-positive
performance alohida ishonchli test qilinmagan.

47.4 Patient-level split yo‘q

Patient ID mavjud emas.

47.5 Probability calibration yo‘q

Threshold 0.993 validation uchun optimal segmentation cutoff.

Bu:

99.3% confidence

degani emas.

47.6 Connected regions ≠ stone count

Connected component soni klinik stone count emas.

47.7 Clinical deployment emas

Loyiha:

research
education
portfolio

uchun.

Tibbiy tashxis vositasi emas.

48. Kelajakdagi project structure

Kelajakda loyiha quyidagicha kengaytirilishi mumkin:

3D CT volume
      ↓
Kidney localization
      ↓
Stone segmentation
      ↓
Voxel spacing
      ↓
Physical measurements
      ↓
Stone volume mm³
      ↓
Diameter mm
      ↓
Longitudinal comparison
      ↓
Clinical decision-support research

Qo‘shimcha modellar:

Attention U-Net;
U-Net++;
nnU-Net;
Swin UNETR;
MONAI SegResNet;
3D U-Net.
49. Ishga tushirish tartibi

Project allaqachon tayyor dataset bilan bo‘lsa:

1. Environment
2. Dataset preparation
3. Train
4. Evaluate
5. Predict
6. Gradio

Asosiy commandlar:

python src/check_environment.py
python src/prepare_data.py \
    --config configs/config.yaml
python src/train.py \
    --config configs/config.yaml
python src/evaluate.py \
    --config configs/config.yaml

Prediction:

python src/predict.py \
    --config configs/config.yaml \
    --image path/to/image.tif

Gradio:

python app/app.py
50. Final loyiha holati

Hozirgi loyiha quyidagi barcha bosqichlarni qamrab oladi:

Dataset download
        ✅
Dataset inspection
        ✅
Image-mask matching
        ✅
Manifest
        ✅
Train/Val/Test split
        ✅
Preprocessing
        ✅
Augmentation
        ✅
2D U-Net
        ✅
Imbalance-aware loss
        ✅
GPU training
        ✅
Best model checkpoint
        ✅
Threshold tuning
        ✅
Postprocessing tuning
        ✅
Final test
        ✅
Global metrics
        ✅
Per-image metrics
        ✅
Area analysis
        ✅
Error analysis
        ✅
Prediction script
        ✅
Visualization
        ✅
Gradio web application
        ✅
README
        ✅
Project documentation
        ✅
51. Yakuniy xulosa

kidney_stone_project_2 oddiy model training misoli emas.

U to‘liq end-to-end segmentation project:

DATA
 ↓
PREPROCESSING
 ↓
MODEL
 ↓
TRAINING
 ↓
VALIDATION
 ↓
HYPERPARAMETER / THRESHOLD TUNING
 ↓
POSTPROCESSING
 ↓
TEST
 ↓
ERROR ANALYSIS
 ↓
INFERENCE
 ↓
WEB APPLICATION

Final model KSSD2025 2D CT tasvirlarda kidney-stone
segmentatsiyasini bajaradi.

Final asosiy natija:

Global Test Dice = 0.6636
Mean Test Dice   = 0.6027

Final deployment konfiguratsiyasi:

Model:
best_2d_unet.pth

Threshold:
0.993

Minimum connected component:
50 pixels

Projectning asosiy yakuniy user flow'i:

CT TIF image
    ↓
Gradio
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

Bu loyiha research, learning va portfolio uchun
to‘liq kidney-stone segmentation pipeline hisoblanadi.