"""
FORAX — Step 4: Train Custom CNN Model
Trains a 5-class forensic image classifier from scratch.
Architecture: Custom 4-block CNN (no transfer learning).
Run: python model_training/4_train_model.py

Output: model_training/forax_forensic_model.h5
        model_training/results/training_curves.png
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt

# ── Check TensorFlow ───────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import Model
    from tensorflow.keras import mixed_precision
    from tensorflow.keras.layers import (
        Conv2D, BatchNormalization, Activation, MaxPooling2D,
        GlobalAveragePooling2D, Dense, Dropout, Input
    )
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
    )
    from tensorflow.keras.optimizers import Adam
    print(f"[OK] TensorFlow {tf.__version__}")
except ImportError:
    print("[FAIL] TensorFlow not installed.")
    print("  Run: pip install tensorflow")
    sys.exit(1)


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# ── CONFIG ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR     = os.path.join(BASE_DIR, "dataset", "val")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH  = os.path.join(BASE_DIR, "forax_forensic_model.h5")
LOG_PATH    = os.path.join(RESULTS_DIR, "training_log.csv")
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")

IMG_SIDE    = int(os.getenv("FORAX_IMG_SIZE", "224"))
IMG_SIZE    = (IMG_SIDE, IMG_SIDE)
_BATCH_ENV  = os.getenv("FORAX_BATCH_SIZE")
BATCH_SIZE  = int(_BATCH_ENV) if _BATCH_ENV else 16
EPOCHS      = int(os.getenv("FORAX_EPOCHS", "30"))
LR          = float(os.getenv("FORAX_LR", "1e-4"))
BACKBONE    = os.getenv("FORAX_BACKBONE", "efficientnetb0").strip().lower()
INITIAL_LR  = float(os.getenv("FORAX_INITIAL_LR", "1e-4"))
FINE_TUNE_EPOCHS = int(os.getenv("FORAX_FINE_TUNE_EPOCHS", "10"))
FINE_TUNE_LR = float(os.getenv("FORAX_FINE_TUNE_LR", "1e-5"))
FINE_TUNE_AT = int(os.getenv("FORAX_FINE_TUNE_AT", "50"))
GPU_BATCH_SIZE = int(os.getenv("FORAX_GPU_BATCH_SIZE", "16"))
AUTO_BATCH_ON_GPU = _as_bool(os.getenv("FORAX_AUTO_BATCH_ON_GPU"), True)
ENABLE_GPU_MEMORY_GROWTH = _as_bool(os.getenv("FORAX_GPU_MEMORY_GROWTH"), True)
ENABLE_MIXED_PRECISION = _as_bool(os.getenv("FORAX_MIXED_PRECISION"), True)
ENABLE_XLA = _as_bool(os.getenv("FORAX_ENABLE_XLA"), False)
USE_CLASS_WEIGHTS = _as_bool(os.getenv("FORAX_USE_CLASS_WEIGHTS"), True)

_CLASSES_ENV = os.getenv("FORAX_CLASSES", "drugs,normal,nsfw,scam,violence,weapons")
CLASSES     = [c.strip() for c in _CLASSES_ENV.split(",") if c.strip()]
if not CLASSES:
    CLASSES = ["drugs", "normal", "nsfw", "violence", "weapons"]
NUM_CLASSES = len(CLASSES)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE — Custom 4-Block CNN
# ══════════════════════════════════════════════════════════════════
def build_model(num_classes, img_size):
    """
    Custom Forensic CNN Architecture
    ─────────────────────────────────
    Block 1: Conv(32)  + BN + ReLU + MaxPool → 112×112
    Block 2: Conv(64)  + BN + ReLU + MaxPool →  56×56
    Block 3: Conv(128) + BN + ReLU + MaxPool →  28×28
    Block 4: Conv(256) + BN + ReLU + MaxPool →  14×14
    Head: GlobalAvgPool → Dense(512) + Dropout(0.5) → Softmax(5)
    """
    model = Sequential([
        Input(shape=(*img_size, 3)),

        # ── Block 1 ────────────────────────────────────────
        Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.1),

        # ── Block 2 ────────────────────────────────────────
        Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.15),

        # ── Block 3 ────────────────────────────────────────
        Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),

        # ── Block 4 ────────────────────────────────────────
        Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        GlobalAveragePooling2D(),

        # ── Classifier Head ────────────────────────────────
        Dense(512, activation='relu', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_initializer='he_normal'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax', dtype='float32')
    ], name='ForaxForensicCNN')

    return model


def build_transfer_model(num_classes, img_size):
    """Build an EfficientNetB0 transfer-learning classifier."""
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(*img_size, 3),
    )
    base_model.trainable = False

    inputs = Input(shape=(*img_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    x = Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)
    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs, outputs, name='ForaxEfficientNetB0')
    return model, base_model


# ══════════════════════════════════════════════════════════════════
# DATA GENERATORS
# ══════════════════════════════════════════════════════════════════
def create_generators(batch_size, use_transfer=False):
    """Create training and validation data generators with augmentation."""

    if use_transfer:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.15,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        classes=CLASSES,
        shuffle=False
    )

    return train_gen, val_gen


def configure_gpu_runtime():
    """Apply optional runtime optimizations when GPUs are available."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("[INFO] Training on CPU (this may take 30-60 minutes)")
        return False

    print(f"[OK] GPUs detected: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu.name}")

    if ENABLE_GPU_MEMORY_GROWTH:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception as e:
                print(f"  [WARN] Could not set memory growth for {gpu.name}: {e}")

    if ENABLE_XLA:
        try:
            tf.config.optimizer.set_jit(True)
            print("  [OK] XLA JIT enabled")
        except Exception as e:
            print(f"  [WARN] Could not enable XLA JIT: {e}")

    if ENABLE_MIXED_PRECISION:
        try:
            mixed_precision.set_global_policy('mixed_float16')
            print("  [OK] Mixed precision enabled")
        except Exception as e:
            print(f"  [WARN] Could not enable mixed precision: {e}")

    return True


def compute_class_weights(train_gen):
    """Compute class weights from generator labels to reduce class imbalance."""
    class_counts = np.bincount(train_gen.classes, minlength=len(train_gen.class_indices))
    total = float(np.sum(class_counts))
    n_classes = len(class_counts)

    class_weights = {}
    for idx, count in enumerate(class_counts):
        if count > 0:
            class_weights[idx] = total / (n_classes * float(count))

    return class_weights, class_counts


# ══════════════════════════════════════════════════════════════════
# PLOT TRAINING CURVES
# ══════════════════════════════════════════════════════════════════
def plot_training_history(history):
    """Save accuracy and loss curves for FYP report."""
    hist = history.history if hasattr(history, 'history') else history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('FORAX Forensic CNN — Training History', fontsize=14, fontweight='bold')

    # Accuracy
    axes[0].plot(hist['accuracy'],     label='Train Accuracy', color='#2196F3', linewidth=2)
    axes[0].plot(hist['val_accuracy'], label='Val Accuracy',   color='#4CAF50', linewidth=2, linestyle='--')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1])

    # Loss
    axes[1].plot(hist['loss'],     label='Train Loss', color='#F44336', linewidth=2)
    axes[1].plot(hist['val_loss'], label='Val Loss',   color='#FF9800', linewidth=2, linestyle='--')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Training curves saved: {output_path}")


# ══════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  FORAX — Custom CNN Training")
    print("=" * 60)
    print(f"  Classes: {', '.join(CLASSES)}")
    print(f"  Config : epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, image={IMG_SIZE[0]}x{IMG_SIZE[1]}")
    print(f"  Backbone: {BACKBONE}")

    if IMG_SIZE[0] <= 0:
        print("[FAIL] FORAX_IMG_SIZE must be a positive integer")
        sys.exit(1)
    if BATCH_SIZE <= 0 or EPOCHS <= 0:
        print("[FAIL] FORAX_BATCH_SIZE and FORAX_EPOCHS must be positive integers")
        sys.exit(1)
    if LR <= 0:
        print("[FAIL] FORAX_LR must be greater than 0")
        sys.exit(1)

    # Validate dataset exists
    if not os.path.exists(TRAIN_DIR):
        print("[FAIL] Training data not found.")
        print("  Run: python model_training/3_prepare_dataset.py")
        sys.exit(1)

    has_gpu = configure_gpu_runtime()

    effective_batch_size = BATCH_SIZE
    if has_gpu and AUTO_BATCH_ON_GPU and _BATCH_ENV is None:
        effective_batch_size = GPU_BATCH_SIZE
        print(f"[OK] Auto GPU batch size set to {effective_batch_size} for VRAM stability")
    print(f"[INFO] Effective batch size: {effective_batch_size}")

    # ── Build Model ────────────────────────────────────────────────
    use_transfer = BACKBONE in ('efficientnetb0', 'efficientnet', 'transfer')
    base_model = None
    print("\n  Building FORAX model...")
    if use_transfer:
        try:
            model, base_model = build_transfer_model(NUM_CLASSES, IMG_SIZE)
            print("  [OK] Using EfficientNetB0 transfer learning backbone")
        except Exception as e:
            print(f"  [WARN] Transfer backbone unavailable, falling back to custom CNN: {e}")
            use_transfer = False

    if not use_transfer:
        model = build_model(NUM_CLASSES, IMG_SIZE)
        print("  [OK] Using custom 4-block CNN backbone")

    initial_lr = INITIAL_LR if use_transfer else LR
    model.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    total_params = model.count_params()
    trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    print(f"\n  Total parameters   : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ── Data Generators ───────────────────────────────────────────
    print("\n  Loading dataset...")
    train_gen, val_gen = create_generators(effective_batch_size, use_transfer=use_transfer)
    print(f"  Train samples : {train_gen.samples}")
    print(f"  Val samples   : {val_gen.samples}")
    print(f"  Classes       : {train_gen.class_indices}")

    class_weight = None
    if USE_CLASS_WEIGHTS:
        class_weight, class_counts = compute_class_weights(train_gen)
        idx_to_label = {v: k for k, v in train_gen.class_indices.items()}
        print("  Class counts  :")
        for idx in range(len(class_counts)):
            label = idx_to_label.get(idx, str(idx))
            print(f"    - {label:<12} {int(class_counts[idx])}")
        print("  [OK] Class weighting enabled")

    # ── Callbacks ─────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        CSVLogger(LOG_PATH)
    ]

    # ── Train ──────────────────────────────────────────────────────
    print(f"\n  Training for up to {EPOCHS} epochs (early stopping enabled)...")
    print("  Best model will be auto-saved to:", MODEL_PATH)
    print(f"{'-'*60}\n")

    histories = []
    initial_epochs = EPOCHS
    fine_tune_epochs = 0
    if use_transfer and FINE_TUNE_EPOCHS > 0:
        initial_epochs = max(1, EPOCHS - FINE_TUNE_EPOCHS)
        fine_tune_epochs = FINE_TUNE_EPOCHS

    history = model.fit(
        train_gen,
        epochs=initial_epochs,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )
    histories.append(history)

    if use_transfer and fine_tune_epochs > 0 and base_model is not None:
        print("\n  Fine-tuning top backbone layers...")
        base_model.trainable = True
        fine_tune_start = max(0, len(base_model.layers) - FINE_TUNE_AT)
        for layer in base_model.layers[:fine_tune_start]:
            layer.trainable = False
        for layer in base_model.layers[fine_tune_start:]:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        model.compile(
            optimizer=Adam(learning_rate=FINE_TUNE_LR),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        fine_tune_history = model.fit(
            train_gen,
            epochs=initial_epochs + fine_tune_epochs,
            initial_epoch=history.epoch[-1] + 1 if history.epoch else initial_epochs,
            validation_data=val_gen,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1
        )
        histories.append(fine_tune_history)

    merged_history = {}
    for h in histories:
        for key, values in h.history.items():
            merged_history.setdefault(key, []).extend(values)

    # ── Results ───────────────────────────────────────────────────
    best_val_acc = max(merged_history['val_accuracy'])
    best_epoch   = merged_history['val_accuracy'].index(best_val_acc) + 1

    print(f"\n{'='*60}")
    print("  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Best Val Accuracy : {best_val_acc*100:.1f}%  (epoch {best_epoch})")
    print(f"  Model saved to    : {MODEL_PATH}")
    print(f"{'-'*60}")

    # Save class indices for inference
    class_idx_path = os.path.join(BASE_DIR, "class_indices.json")
    with open(class_idx_path, 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"  Class indices     : {class_idx_path}")

    model_config = {
        "backbone": "efficientnetb0" if use_transfer else "custom_cnn",
        "img_size": IMG_SIDE,
        "batch_size": effective_batch_size,
        "preprocessing": "efficientnet" if use_transfer else "rescale_255",
        "classes": CLASSES,
        "fine_tune_epochs": fine_tune_epochs,
        "fine_tune_at": FINE_TUNE_AT if use_transfer else 0,
        "mixed_precision": bool(has_gpu and ENABLE_MIXED_PRECISION),
        "class_weights": bool(USE_CLASS_WEIGHTS),
    }
    with open(MODEL_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(model_config, f, indent=2)
    print(f"  Model config      : {MODEL_CONFIG_PATH}")

    # Plot
    plot_training_history(merged_history)

    print(f"\n  Next: python model_training/5_evaluate_model.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
