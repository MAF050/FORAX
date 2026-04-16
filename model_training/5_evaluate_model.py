"""
FORAX — Step 5: Evaluate Model & Generate FYP Report Assets
Loads the trained model and generates:
  - Confusion matrix (PNG)
  - Classification report (TXT + console)
  - Per-class accuracy bar chart (PNG)
Run: python model_training/5_evaluate_model.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.efficientnet import preprocess_input
except ImportError:
    print("[FAIL] TensorFlow not installed.")
    sys.exit(1)

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        accuracy_score, f1_score
    )
except ImportError:
    print("[FAIL] scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

# ── CONFIG ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VAL_DIR     = os.path.join(BASE_DIR, "dataset", "val")
MODEL_PATH  = os.path.join(BASE_DIR, "forax_forensic_model.h5")
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

IMG_SIDE    = int(os.getenv("FORAX_IMG_SIZE", "224"))
IMG_SIZE    = (IMG_SIDE, IMG_SIDE)
BATCH_SIZE  = int(os.getenv("FORAX_BATCH_SIZE", "32"))
_CLASSES_ENV = os.getenv("FORAX_CLASSES", "drugs,normal,nsfw,scam,violence,weapons")
CLASSES     = [c.strip() for c in _CLASSES_ENV.split(",") if c.strip()]
if not CLASSES:
    CLASSES = ["drugs", "normal", "nsfw", "violence", "weapons"]


def load_model_config():
    if os.path.exists(MODEL_CONFIG_PATH):
        with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


MODEL_CONFIG = load_model_config()

if MODEL_CONFIG.get('classes'):
    CLASSES = [c.strip() for c in MODEL_CONFIG['classes'] if str(c).strip()]
else:
    CLASSES = [c.strip() for c in _CLASSES_ENV.split(",") if c.strip()]
if not CLASSES:
    CLASSES = ["drugs", "normal", "nsfw", "violence", "weapons"]

IMG_SIDE = int(MODEL_CONFIG.get('img_size', IMG_SIDE))
IMG_SIZE = (IMG_SIDE, IMG_SIDE)

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_val_data():
    """Load validation dataset."""
    config = load_model_config()
    preprocessing = config.get('preprocessing', 'rescale_255')
    if preprocessing == 'efficientnet':
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    gen = datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )
    return gen


def plot_confusion_matrix(cm, classes):
    """Save confusion matrix as a high-quality PNG."""
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=[c.upper() for c in classes],
        yticklabels=[c.upper() for c in classes],
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label',      fontsize=12, fontweight='bold')
    ax.set_title('FORAX Forensic CNN — Confusion Matrix\n(Normalized)', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Confusion matrix saved: {out_path}")


def plot_per_class_accuracy(report_dict, classes):
    """Bar chart of per-class F1 scores."""
    f1_scores = [report_dict.get(cls, {}).get('f1-score', 0) for cls in classes]
    colors = ['#FF6B6B' if f < 0.7 else '#4ECDC4' if f < 0.85 else '#2ECC71' for f in f1_scores]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([c.upper() for c in classes], f1_scores, color=colors, edgecolor='white', linewidth=1.2)

    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Forensic Category', fontsize=12)
    ax.set_ylabel('F1 Score',          fontsize=12)
    ax.set_title('FORAX Forensic CNN — Per-Class F1 Score', fontsize=13, fontweight='bold')
    ax.axhline(y=0.70, color='orange', linestyle='--', alpha=0.6, label='Acceptable threshold (0.70)')
    ax.axhline(y=0.85, color='green',  linestyle='--', alpha=0.6, label='Good threshold (0.85)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "per_class_f1.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Per-class F1 chart saved: {out_path}")


def save_report_txt(report_str, accuracy, macro_f1):
    """Save full classification report to text file."""
    out_path = os.path.join(RESULTS_DIR, "classification_report.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  FORAX FORENSIC CNN — CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Overall Accuracy : {accuracy*100:.2f}%\n")
        f.write(f"  Macro F1 Score   : {macro_f1:.4f}\n\n")
        f.write("  Classes: drugs | normal | nsfw | scam | violence | weapons\n")
        f.write("  Image size: 224x224 px\n")
        f.write("  Architecture: EfficientNetB0 (Transfer Learning)\n\n")
        f.write("-" * 60 + "\n")
        f.write(report_str)
        f.write("\n" + "=" * 60 + "\n")
    print(f"  [OK] Classification report: {out_path}")
    return out_path


def main():
    print("=" * 60)
    print("  FORAX - Model Evaluation")
    print("=" * 60)
    print(f"  Classes: {', '.join(CLASSES)}")
    print(f"  Config : batch={BATCH_SIZE}, image={IMG_SIZE[0]}x{IMG_SIZE[1]}")

    if IMG_SIZE[0] <= 0:
        print("[FAIL] FORAX_IMG_SIZE must be a positive integer")
        sys.exit(1)
    if BATCH_SIZE <= 0:
        print("[FAIL] FORAX_BATCH_SIZE must be a positive integer")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"[FAIL] Trained model not found: {MODEL_PATH}")
        print("  Run: python model_training/4_train_model.py")
        sys.exit(1)

    if not os.path.exists(VAL_DIR):
        print("[FAIL] Validation data not found.")
        print("  Run: python model_training/3_prepare_dataset.py")
        sys.exit(1)

    # ── Load Model ────────────────────────────────────────────────
    print(f"\n  Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    print(f"  [OK] Model loaded")

    # ── Load Validation Data ──────────────────────────────────────
    print("  Loading validation data...")
    val_gen = load_val_data()
    print(f"  Samples: {val_gen.samples}  |  Batches: {len(val_gen)}")

    # ── Predictions ───────────────────────────────────────────────
    print("\n  Running predictions (this may take a few minutes)...")
    val_gen.reset()
    y_pred_probs = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    # ── Metrics ───────────────────────────────────────────────────
    accuracy  = accuracy_score(y_true, y_pred)
    macro_f1  = f1_score(y_true, y_pred, average='macro')
    cm        = confusion_matrix(y_true, y_pred)
    report_str = classification_report(y_true, y_pred, target_names=CLASSES)
    report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)

    # ── Print Results ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Overall Accuracy  : {accuracy*100:.2f}%")
    print(f"  Macro F1 Score    : {macro_f1:.4f}")
    print(f"\n  Per-Class Report:")
    print("-" * 60)
    print(report_str)

    # ── Per-class summary ─────────────────────────────────────────
    print(f"{'='*60}")
    print(f"  {'Class':<12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print("  " + "-" * 46)
    for cls in CLASSES:
        p  = report_dict.get(cls, {}).get('precision', 0)
        r  = report_dict.get(cls, {}).get('recall',    0)
        f1 = report_dict.get(cls, {}).get('f1-score',  0)
        bar = '#' * int(f1 * 20) + '.' * (20 - int(f1 * 20))
        print(f"  {cls:<12}  {p:>10.2%}  {r:>8.2%}  {f1:>8.2%}  {bar}")
    print(f"{'='*60}")

    # ── Save Artifacts ────────────────────────────────────────────
    print("\n  Saving FYP report assets...")
    plot_confusion_matrix(cm, CLASSES)
    plot_per_class_accuracy(report_dict, CLASSES)
    save_report_txt(report_str, accuracy, macro_f1)

    # ── Final Summary ─────────────────────────────────────────────
    grade = "EXCELLENT" if accuracy > 0.85 else "GOOD" if accuracy > 0.75 else "ACCEPTABLE"
    print(f"\n{'='*60}")
    print(f"  GRADE: {grade}  ({accuracy*100:.1f}%)")
    print("-" * 60)
    print(f"  Assets saved in: model_training/results/")
    print(f"    - confusion_matrix.png  (for FYP Chapter 4)")
    print(f"    - per_class_f1.png      (for FYP Chapter 4)")
    print(f"    - classification_report.txt")
    print(f"{'='*60}")
    print(f"\n  FORAX is ready. Start the app:")
    print(f"    python app.py")
    print(f"  (Local model will be used automatically - no API needed)")


if __name__ == "__main__":
    main()
