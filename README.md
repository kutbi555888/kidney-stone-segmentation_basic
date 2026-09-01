<div align="center">

# 🩺🪨 Kidney Stone Segmentation

### 🧠 **2D U-Net** · 🗂️ **KSSD2025** · 🖼️ **512 × 512** · 🚀 **Gradio**

> 🧬 **CT tasvirlarda kidney-stone hududlarini avtomatik segmentatsiya qilish loyihasi**

</div>

---


## 🩺 1. Loyiha haqida


Ushbu loyiha 2D CT tasvirlarda kidney stone hududlarini
avtomatik segmentatsiya qilish uchun ishlab chiqilgan.

#### 🎯 Asosiy vazifa:

> CT tasvirdagi kidney-stone pixellarini aniqlash va
> binary segmentation mask yaratish.

Model sifatida **2D U-Net** arxitekturasi ishlatilgan.



---

## 🗂️ 2. Dataset


Loyihada **KSSD2025 Kidney Stone Segmentation Dataset**
ishlatilgan.

#### 📦 Dataset tarkibi:

- 838 ta 2D axial CT TIF tasvir
- 838 ta binary stone mask
- barcha masklarda stone foreground mavjud
- image va label fayllari alohida saqlangan

#### 🗃️ Dataset strukturasi:

```text
data/raw/kssd2025/data/
├── image/
│   ├── 1.tif
│   ├── 10.tif
│   └── ...
└── label/
    ├── 1.tif
    ├── 10.tif
    └── ...

```

KSSD2025 2D dataset bo‘lgani sabab ushbu loyihada
haqiqiy 3D stone volume hisoblanmaydi.

Pixel spacing fizik metadata pipeline'da ishlatilmagani uchun
maydon natijalari pixel birlikda beriladi.


---

## ✂️ 3. Train / Validation / Test split


#### ✂️ Dataset quyidagicha bo‘lindi:

**Total      : 838 images**
**Train      : 586 images**
**Validation : 126 images**
**Test       : 126 images**

Split image-level asosida amalga oshirilgan.

Fayl nomlarida patient ID mavjud bo‘lmagani sabab
patient-level split amalga oshirib bo‘lmadi.

Shuning uchun bir patientga tegishli bir nechta slice mavjud
bo‘lsa, patient-level leakage ehtimolini to‘liq inkor qilib
bo‘lmaydi.


---

## 🧼 4. Preprocessing


#### 🧼 CT tasvirlarga quyidagi preprocessing qo‘llanadi:

- 🧼 TIF rasmni o‘qish
- 🧼 Grayscale formatga o‘tkazish
- 🧼 0.5 va 99.5 percentile clipping
- 🧼 0–1 oralig‘iga normalization
- 🧼 512 × 512 o‘lchamga resize

#### 🎭 Masklar:

- 🧼 binary formatga aylantiriladi
- 🧼 nearest-neighbor interpolation bilan resize qilinadi

#### 🔁 Training vaqtida konservativ augmentation ishlatiladi:

- 🧼 horizontal flip

Anatomik jihatdan noto‘g‘ri transformatsiyalarni kamaytirish
uchun vertical flip va katta rotationlardan foydalanilmaydi.


---

## 🧠 5. Model


#### 🧠 Asosiy model:

2D U-Net

#### ⚙️ Konfiguratsiya:

Input channels : 1
Output channels: 1

Channels:
16
32
64
128

Downsampling:
3 bosqich

Image size:
512 × 512

Kichik kidney-stone regionlarini saqlab qolish uchun
juda ko‘p downsampling ishlatilmadi.


---

## ⚖️ 6. Loss function


Kidney-stone foreground tasvirning juda kichik qismini egallaydi.

#### 📉 Dataset analizida foreground ratio taxminan:

> ⭐ **0.135%**

atrofida bo‘lgan.

Shu sabab oddiy BCE loss kichik stone regionlarini yetarlicha
o‘rganmagan.

#### ⚖️ Final loss:

Weighted BCE + Tversky Loss

#### 🧮 Taxminiy kombinatsiya:

0.4 × Weighted BCE
+
0.6 × Tversky Loss

#### ➕ Weighted BCE:

pos_weight = 50

#### 🎯 Tversky:

alpha = 0.3
beta  = 0.7

Bu konfiguratsiya false-negative xatolarni kamaytirishga
yordam beradi.


---

## 🏋️ 7. Training


#### 🏋️ Final training konfiguratsiyasi:

**Epochs        : 25**
**Batch size    : 4**
**Learning rate : 0.0002**
**Weight decay  : 0.00001**
**AMP           : True**
**Input size    : 512 × 512**

Training Google Colab GPU muhitida bajarilgan.

#### 💾 Final model:

> ⭐ **models/best_2d_unet.pth**

Validation Dice bo‘yicha eng yaxshi checkpoint saqlangan.


---

## 🎚️ 8. Threshold tuning


#### 🔧 Default:

**threshold = 0.5**

ushbu model uchun optimal bo‘lmagan.

Threshold faqat validation dataset orqali tanlangan.

#### 🔬 Validation search natijasida:

> ⭐ **Final threshold = 0.993**

tanlandi.

### ⚠️ Muhim:

> ⚠️ **0.993 ni 99.3% kalibrlangan klinik ishonchlilik deb**  \
> **talqin qilish mumkin emas.**

Model output probability qiymatlari kalibrlanmagan.


---

## 🧹 9. Postprocessing


Predictiondan keyin kichik connected componentlar olib
tashlanadi.

Postprocessing parametri ham faqat validation datasetda
tanlangan.

#### ✅ Final qiymat:

> ⭐ **min_component_pixels = 50**

#### 🔄 Pipeline:
```text

Model logits
    ↓
Sigmoid
    ↓
Threshold = 0.993
    ↓
Binary mask
    ↓
< 50 pixel componentlarni olib tashlash
    ↓
Final segmentation mask
```

---

## ✅ 10. Validation natijalari


#### ✅ Final validation postprocessing konfiguratsiyasida:

**Global Dice      : 0.6520**
**Global IoU       : 0.4837**
**Global Precision : 0.6122**
**Global Recall    : 0.6974**

---

## 📊 11. Final Test natijalari


#### 🧪 Final test set:

> ⭐ **126 images**
#### 📏 Mean per-image metrics

Har bir image uchun metric alohida hisoblanib,
keyin o‘rtacha olinadi.

**Mean Dice       : 0.6027**
**Mean IoU        : 0.4478**
**Mean Precision  : 0.5596**
**Mean Recall     : 0.6887**
#### 🌐 Global metrics

Barcha test pixellar birgalikda hisoblanadi.

**Global Dice       : 0.6636**
**Global IoU        : 0.4966**
**Global Precision  : 0.6174**
**Global Recall     : 0.7173**
#### 📐 Area
#### 📐 Mean absolute area error:
90.94 pixels

Global va mean per-image metrikalar bir xil metric emas.

Global metric kattaroq stone regionlarga ko‘proq vazn beradi,
mean per-image metric esa har bir tasvirga teng vazn beradi.


---

## 🔎 12. Best / Median / Worst case analysis


Test predictionlar Dice bo‘yicha analiz qilindi.

#### 🏆 Eng yaxshi misollardan biri:

`312.tif`

**Dice      : 0.8616**
**Precision : 0.8431**
**Recall    : 0.8809**

#### 📌 O‘rtacha case:

`375.tif`

**Dice      : 0.6061**
**Precision : 0.5466**
**Recall    : 0.6801**

#### ⚠️ Qiyin case misoli:

`880.tif`

**Dice      : 0.1579**
**Precision : 0.2542**
**Recall    : 0.1145**

#### 🖼️ Visualization:

`results/figures/best_median_worst_cases.png`

Bu analysis model kichik, kontrasti past yoki murakkab
stone regionlarda xato qilishi mumkinligini ko‘rsatadi.


---

## 🔮 13. Prediction


#### 🔮 Yangi TIF image uchun prediction:

```bash
python src/predict.py \
    --config configs/config.yaml \
    --image path/to/image.tif
```
#### 💡 Misol:

```bash
python src/predict.py \
    --config configs/config.yaml \
    --image data/raw/kssd2025/data/image/312.tif
```
#### 📤 Prediction natijalari:

`results/predictions/`

ichiga saqlanadi.

#### 💡 Masalan:

`312_mask.png`
`312_probability.png`
`312_prediction.png`

---

## 🧾 14. Prediction output


#### 🧾 Prediction pipeline quyidagilarni beradi:

- ✨ Final segmentation mask
- ✨ Probability map
- ✨ Stone overlay
- ✨ Predicted stone area
- ✨ Estimated connected regions
- ✨ Region areas

#### 💡 Misol:

`312.tif`

Predicted stone area:
> ⭐ **816 pixels**

Estimated connected regions:
1

> ⚠️ **Estimated connected regions klinik jihatdan aniq**  \
> **kidney-stone soni degani emas.**

> ⚠️ Dataset masklarida juda ko‘p kichik connected componentlar  \
> mavjud bo‘lgani sabab component count klinik stone count  \
> sifatida ishlatilmaydi.


---

## 🚀 15. Gradio Web App


Loyihada Gradio interfeys mavjud.

#### ▶️ Ishga tushirish:

```bash
python app/app.py
```

Gradio orqali foydalanuvchi TIF CT image yuklaydi.

#### 🔄 Pipeline:
```text

CT upload
    ↓
Preprocessing
    ↓
2D U-Net
    ↓
Sigmoid
    ↓
Threshold = 0.993
    ↓
Postprocessing = 50 pixels
    ↓
Final mask
    ↓
Overlay + quantitative output

```

#### 🖥️ Gradio quyidagilarni ko‘rsatadi:

- ✨ Preprocessed CT
- ✨ Probability map
- ✨ Final segmentation mask
- ✨ Stone overlay
- ✨ Predicted pixel area
- ✨ Estimated connected regions

---

## 📁 16. Project structure

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
│   ├── processed/
│   └── splits/
│
├── models/
│   ├── best_2d_unet.pth
│   └── last_checkpoint.pth
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
├── requirements.txt
├── README.md
└── PROJECT_STRUCTURE.md
```

---

## 🧩 17. Asosiy fayllar


### ⚙️ `configs/config.yaml`


#### ⚙️ Project konfiguratsiyasi:

- ⚙️ image size
- ⚙️ training parameters
- ⚙️ model architecture
- ⚙️ **threshold**
- ⚙️ postprocessing
- ⚙️ pathlar

#### ⭐ Final muhim qiymatlar:

**image_size            = 512 × 512**
**threshold             = 0.993**
**min_component_pixels  = 50**

### 🧼 `src/dataset.py`

- 🧩 TIF image o‘qish
- 🧩 mask o‘qish
- 🧩 normalization
- 🧩 resize
- 🧩 augmentation
- 🧩 tensor yaratish

### 🧠 `src/model.py`


2D U-Net modelini yaratadi.


### ⚖️ `src/losses.py`


Weighted BCE + Tversky lossni hisoblaydi.


### 🏋️ `src/train.py`


Model training pipeline.


### 📊 `src/evaluate.py`


Final test evaluation.

#### 📊 Hisoblaydi:

- 📊 **Mean Dice**
- 📊 **Mean IoU**
- 📊 **Mean Precision**
- 📊 **Mean Recall**
- 📊 **Global Dice**
- 📊 **Global IoU**
- 📊 **Global Precision**
- 📊 **Global Recall**
- 📊 Area MAE

### 🔮 `src/predict.py`


Yangi CT image uchun inference qiladi.


### 🚀 `app/app.py`


Gradio web interface.


---

## ⚠️ 18. Loyihaning cheklovlari


#### ⚠️ Ushbu modelning bir nechta muhim cheklovlari mavjud.


### 🔸 1. 2D dataset


KSSD2025 2D TIF dataset.

Shuning uchun model:

- ⚠️ **3D stone volume**
- ⚠️ **stone volume in mm³**

hisoblamaydi.


### 🔸 2. Physical spacing


Pipeline physical pixel spacing metadata ishlatmaydi.

Shuning uchun:

- ⚠️ **Area = pixels**

va:

- ⚠️ **Area ≠ mm²**

### 🔸 3. Negative cases


Datasetdagi barcha foydalanilgan masklarda foreground mavjud.

Shuning uchun stone mavjud bo‘lmagan CT tasvirlarda modelning
false-positive xatti-harakati to‘liq baholanmagan.


### 🔸 4. Patient-level split


Patient ID mavjud bo‘lmagani sabab image-level split ishlatilgan.


### 🔸 5. Clinical use


> ⚠️ Model klinik tashxis tizimi emas.

U tadqiqot, o‘quv va portfolio maqsadida ishlab chiqilgan.


---

## 🔭 19. Kelajakdagi yaxshilanishlar


#### 🔭 Keyingi versiyalarda:

- 🔹 patient-level split
- 🔹 stone-negative CT cases
- 🔹 larger dataset
- 🔹 attention U-Net
- 🔹 nnU-Net
- 🔹 MONAI architectures
- 🔹 Dice + Focal/Tversky loss tuning
- 🔹 pretrained encoder
- 🔹 cross-validation
- 🔹 probability calibration
- 🔹 CT physical spacing
- 🔹 3D segmentation
- 🔹 stone volume in mm³

kabi imkoniyatlarni qo‘shish mumkin.


---

## 🏁 20. Final pipeline

```text
KSSD2025 CT image
        ↓
Data preparation
        ↓
Train / Validation / Test
        ↓
512 × 512 preprocessing
        ↓
2D U-Net
        ↓
Weighted BCE + Tversky
        ↓
Best checkpoint
        ↓
Validation threshold tuning
        ↓
Threshold = 0.993
        ↓
Validation postprocessing tuning
        ↓
Min component = 50 pixels
        ↓
Final test evaluation
        ↓
Prediction
        ↓
Gradio application
```

### 🏁 Final result

```text
Global Test Dice       : 0.6636
Global Test IoU        : 0.4966
Global Test Precision  : 0.6174
Global Test Recall     : 0.7173

Mean Test Dice         : 0.6027
Mean Test IoU          : 0.4478

```

Ushbu loyiha kidney-stone segmentation uchun to‘liq
end-to-end deep-learning pipeline hisoblanadi:

dataset → preprocessing → training → validation tuning →
test evaluation → inference → visualization → Gradio.

---

<div align="center">

### 🏁 **END-TO-END PIPELINE**

🗂️ Dataset → 🧼 Preprocessing → 🧠 U-Net → 🏋️ Training → 🎚️ Tuning → 📊 Evaluation → 🔮 Prediction → 🚀 Gradio

</div>
