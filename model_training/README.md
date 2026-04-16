# FORAX Custom Forensic CNN — Training Guide

Train a 5-class forensic image classifier using Kaggle datasets.
No external API needed after training — 100% offline inference.

---

## Prerequisites
1. Python 3.10+ with FORAX venv activated
2. A free Kaggle account → https://www.kaggle.com
3. ~4 GB free disk space

---

## Step-by-Step

### Step 1 — Get Your Kaggle API Key
1. Log in to https://www.kaggle.com
2. Click your profile icon → **Settings**
3. Scroll to **API** section → Click **"Create New Token"**
4. This downloads `kaggle.json` to your Downloads folder
5. Move it to: `C:\Users\<YourName>\.kaggle\kaggle.json`

### Step 2 — Verify Setup
```bash
python model_training/1_setup_kaggle.py
```

### Step 3 — Download Datasets (~3 GB total)
```bash
python model_training/2_download_datasets.py
```

Optional controls (recommended on slow networks):

- Download only certain classes:
  - `FORAX_CLASSES=normal,weapons,violence,drugs,nsfw`
- Cap how many raw images are kept per class (default `5000`):
  - `FORAX_MAX_RAW_PER_CLASS=200`
- NSFW partial mode (avoids downloading the full multi-GB archive):
  - `FORAX_NSFW_PARTIAL=1` with `FORAX_MAX_RAW_PER_CLASS=<N>`

PowerShell example (download only 200 NSFW images):

```powershell
$env:FORAX_CLASSES='nsfw'
$env:FORAX_NSFW_PARTIAL='1'
$env:FORAX_MAX_RAW_PER_CLASS='200'
python model_training/2_download_datasets.py
```

### Step 4 — Prepare Dataset
```bash
python model_training/3_prepare_dataset.py
```

Useful controls for low-data classes:

- Keep more validated raw images per class (default `5000`):
  - `FORAX_MAX_PER_CLASS=6000`
- Auto-expand train split with augmentation to a target count (default `2500`):
  - `FORAX_TARGET_TRAIN_PER_CLASS=3000`
  - `FORAX_AUGMENT_LOW_CLASSES=1`
  - `FORAX_MAX_AUG_PER_IMAGE=6`

PowerShell example:

```powershell
$env:FORAX_MAX_PER_CLASS='6000'
$env:FORAX_TARGET_TRAIN_PER_CLASS='3000'
$env:FORAX_AUGMENT_LOW_CLASSES='1'
python model_training/3_prepare_dataset.py
```

### Step 5 — Train the Model (~30 min CPU / ~10 min GPU)
```bash
python model_training/4_train_model.py
```

By default, the trainer now uses an EfficientNetB0 transfer-learning backbone and writes
`model_training/model_config.json` so evaluation and app inference stay in sync.

Optional accuracy controls:

- Backbone selection:
  - `FORAX_BACKBONE=efficientnetb0` (default, recommended for better accuracy)
  - `FORAX_BACKBONE=custom_cnn` (original architecture)
- Fine-tuning controls:
  - `FORAX_INITIAL_LR=1e-4`
  - `FORAX_FINE_TUNE_EPOCHS=8`
  - `FORAX_FINE_TUNE_LR=1e-5`
  - `FORAX_FINE_TUNE_AT=40`

GPU controls (recommended for RTX 4050 6GB):

- Auto use a safer GPU batch size:
  - `FORAX_AUTO_BATCH_ON_GPU=1`
  - `FORAX_GPU_BATCH_SIZE=16`
- Enable mixed precision (faster on NVIDIA RTX):
  - `FORAX_MIXED_PRECISION=1`
- Handle class imbalance automatically:
  - `FORAX_USE_CLASS_WEIGHTS=1`

PowerShell example:

```powershell
$env:FORAX_AUTO_BATCH_ON_GPU='1'
$env:FORAX_GPU_BATCH_SIZE='16'
$env:FORAX_MIXED_PRECISION='1'
$env:FORAX_USE_CLASS_WEIGHTS='1'
$env:FORAX_EPOCHS='35'
python model_training/4_train_model.py
```

If GPU is not detected in native Windows TensorFlow, use WSL2 + NVIDIA CUDA drivers for full GPU training support.

To push accuracy higher, retrain with more balanced data per class and keep NSFW as large as possible.

### Step 6 — Evaluate & Generate Report
```bash
python model_training/5_evaluate_model.py
```

### Step 7 — Run FORAX (uses local model automatically)
```bash
python app.py
```

---

## Text/NLP Training (Threat, Harassment, Scam, Fraud)

### Step 6 — Download Text Datasets (Kaggle)
```bash
python model_training/6_download_text_datasets.py
```

### Step 7 — Train Text Model (CSV sources)
Use one or more CSV sources. Format:
`class:csv_path:text_column[:label_column[:label_value]]`

Example:
```bash
python model_training/7_train_text_model.py \
  --source scam:model_training/text_data/raw/scam/fake_job_postings.csv:description \
  --source harassment:model_training/text_data/raw/harassment/harassment.csv:description
```

After training, the NLP model is loaded automatically by FORAX.

---

## Tabular Training (Fraud/Scam CSV)

### Step 8 — Train Tabular Model
```bash
python model_training/8_train_tabular_model.py --csv D:/path/to/creditcard.csv --label-col Class
```

This saves a tabular model for external or future integration.

---

## Train With Your Own Labeled Dataset (No Kaggle)

If you already have your own labeled images, you can skip Kaggle download
and train directly using one command.

Expected folder layout:

```text
<your_dataset>/
  normal/
  weapons/
  violence/
  drugs/
  nsfw/
```

Run:

```bash
python model_training/train_own_model.py --source D:/path/to/your_dataset
```

Optional tuning:

```bash
python model_training/train_own_model.py --source D:/path/to/your_dataset --epochs 20 --batch-size 16 --img-size 224 --val-split 0.2
```

---

## Datasets Used

| Class | Kaggle Dataset | Description |
|-------|---------------|-------------|
| `normal` | `puneet6060/intel-image-classification` | Buildings, nature, streets |
| `weapons` | `iqmansingh/guns-knives-object-detection` | Guns + knives (weapon imagery) |
| `violence` | `abdulmananraja/real-life-violence-situations` | Violence vs. non-violence imagery |
| `drugs` | `vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images` | Pills/capsules/tablets (proxy) |
| `nsfw` | `jjeevanprakash/nsfw-detection` | Suspicious/NSFW content (academic dataset) |

---

## Output Files
```
model_training/
  forax_forensic_model.h5        ← Trained model (loaded by FORAX)
  results/
    training_curves.png          ← Accuracy/Loss plots (for FYP report)
    confusion_matrix.png         ← Per-class performance
    classification_report.txt    ← Precision, Recall, F1
```

---

## Classes & Risk Mapping

| Model Class | FORAX Risk Level |
|-------------|-----------------|
| `normal` | LOW |
| `drugs` | MED |
| `nsfw` | MED |
| `violence` | HIGH |
| `weapons` | HIGH |
