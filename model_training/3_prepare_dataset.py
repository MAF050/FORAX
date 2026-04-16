"""
FORAX — Step 3: Prepare Dataset
Organizes raw downloaded images into train/validation splits.
Resizes all images to 224×224. Applies basic quality filtering.
Run: python model_training/3_prepare_dataset.py

Output structure:
  dataset/
    train/
      normal/      weapons/      violence/      drugs/      nsfw/
    val/
      normal/      weapons/      violence/      drugs/      nsfw/
"""

import os
import sys
import random
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageOps
except ImportError:
    print("[FAIL] Pillow not installed. Run: pip install Pillow")
    sys.exit(1)


def _as_bool(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

# ── CONFIG ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, "dataset", "raw")
TRAIN_DIR   = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR     = os.path.join(BASE_DIR, "dataset", "val")

IMG_SIDE    = int(os.getenv("FORAX_IMG_SIZE", "224"))
IMG_SIZE    = (IMG_SIDE, IMG_SIDE)
VAL_SPLIT   = float(os.getenv("FORAX_VAL_SPLIT", "0.20"))  # 20% validation
MAX_PER_CLASS = int(os.getenv("FORAX_MAX_PER_CLASS", "5000"))
TARGET_TRAIN_PER_CLASS = int(os.getenv("FORAX_TARGET_TRAIN_PER_CLASS", "2500"))
AUGMENT_LOW_CLASSES = _as_bool(os.getenv("FORAX_AUGMENT_LOW_CLASSES"), True)
MAX_AUG_PER_IMAGE = int(os.getenv("FORAX_MAX_AUG_PER_IMAGE", "6"))
SEED        = int(os.getenv("FORAX_SEED", "42"))

_CLASSES_ENV = os.getenv("FORAX_CLASSES", "normal,weapons,violence,drugs,nsfw,scam")
CLASSES = [c.strip() for c in _CLASSES_ENV.split(",") if c.strip()]
if not CLASSES:
    CLASSES = ["normal", "weapons", "violence", "drugs", "nsfw"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_valid_image(path):
    """Check if a file is a valid, readable image."""
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def augment_image(img, rng):
    """Apply light random augmentation to generate diverse training variants."""
    out = img.copy()

    if rng.random() < 0.5:
        out = ImageOps.mirror(out)
    if rng.random() < 0.15:
        out = ImageOps.flip(out)

    angle = rng.uniform(-20, 20)
    out = out.rotate(angle, resample=Image.BICUBIC, fillcolor=(0, 0, 0))

    dx = int(rng.uniform(-0.08, 0.08) * IMG_SIDE)
    dy = int(rng.uniform(-0.08, 0.08) * IMG_SIDE)
    out = out.transform(
        out.size,
        Image.AFFINE,
        (1, 0, dx, 0, 1, dy),
        resample=Image.BICUBIC,
        fillcolor=(0, 0, 0)
    )

    out = ImageEnhance.Brightness(out).enhance(rng.uniform(0.80, 1.20))
    out = ImageEnhance.Contrast(out).enhance(rng.uniform(0.80, 1.25))
    out = ImageEnhance.Color(out).enhance(rng.uniform(0.80, 1.20))

    return out


def expand_train_split(class_name, train_class_dir, target_count):
    """Generate augmented train images until target count is reached."""
    if target_count <= 0:
        return 0

    source_images = []
    for fname in os.listdir(train_class_dir):
        if Path(fname).suffix.lower() in IMAGE_EXTS:
            source_images.append(os.path.join(train_class_dir, fname))

    if not source_images:
        return 0

    current_count = len(source_images)
    if current_count >= target_count:
        return 0

    needed = target_count - current_count
    max_possible = len(source_images) * MAX_AUG_PER_IMAGE
    if needed > max_possible:
        print(f"  [WARN] {class_name}: target needs {needed} aug images but max is {max_possible} with current cap")
        needed = max_possible

    rng = random.Random(SEED + sum(ord(c) for c in class_name) * 17)
    usage = {path: 0 for path in source_images}
    active = source_images[:]
    generated = 0

    while generated < needed and active:
        src = rng.choice(active)
        if usage[src] >= MAX_AUG_PER_IMAGE:
            active.remove(src)
            continue

        try:
            img = Image.open(src).convert("RGB")
            aug = augment_image(img, rng)
            out_name = f"{class_name}_aug_{generated:05d}.jpg"
            aug.save(os.path.join(train_class_dir, out_name), "JPEG", quality=90)
            generated += 1
            usage[src] += 1
        except Exception:
            usage[src] = MAX_AUG_PER_IMAGE
            if src in active:
                active.remove(src)

    return generated


def prepare_class(class_name):
    """Process one class: filter, resize, split into train/val."""
    src_dir = os.path.join(RAW_DIR, class_name)

    if not os.path.exists(src_dir):
        print(f"  [SKIP] {class_name}/ not found in dataset/raw/")
        return 0, 0, 0

    # Collect all image paths
    all_images = []
    for fname in os.listdir(src_dir):
        if Path(fname).suffix.lower() in IMAGE_EXTS:
            all_images.append(os.path.join(src_dir, fname))

    if not all_images:
        print(f"  [SKIP] {class_name}/ — no images found")
        return 0, 0, 0

    # Shuffle deterministically
    rng = random.Random(SEED + sum(ord(c) for c in class_name))
    rng.shuffle(all_images)

    # Filter valid images and cap
    valid_images = []
    print(f"  Validating {len(all_images)} images for class '{class_name}'...")
    for p in all_images:
        if is_valid_image(p):
            valid_images.append(p)
        if len(valid_images) >= MAX_PER_CLASS:
            break

    # Split
    split_idx   = int(len(valid_images) * (1 - VAL_SPLIT))
    train_imgs  = valid_images[:split_idx]
    val_imgs    = valid_images[split_idx:]

    # Create output dirs
    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    val_class_dir   = os.path.join(VAL_DIR,   class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir,   exist_ok=True)

    def process_and_save(img_list, dest_dir, split_name):
        saved = 0
        for i, src in enumerate(img_list):
            try:
                img = Image.open(src).convert("RGB")
                img = img.resize(IMG_SIZE, Image.LANCZOS)
                fname = f"{class_name}_{split_name}_{i:04d}.jpg"
                img.save(os.path.join(dest_dir, fname), "JPEG", quality=90)
                saved += 1
            except Exception:
                pass
        return saved

    train_saved = process_and_save(train_imgs, train_class_dir, "train")
    val_saved   = process_and_save(val_imgs,   val_class_dir,   "val")

    aug_saved = 0
    if AUGMENT_LOW_CLASSES:
        aug_saved = expand_train_split(class_name, train_class_dir, TARGET_TRAIN_PER_CLASS)

    return train_saved + aug_saved, val_saved, aug_saved


def print_dataset_summary():
    """Print final dataset statistics."""
    print(f"\n{'='*60}")
    print("  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Class':<15} {'Train':>7} {'Val':>7} {'Total':>7}")
    print(f"  {'-'*42}")

    grand_train = grand_val = 0
    for cls in CLASSES:
        t = len(os.listdir(os.path.join(TRAIN_DIR, cls))) if os.path.exists(os.path.join(TRAIN_DIR, cls)) else 0
        v = len(os.listdir(os.path.join(VAL_DIR,   cls))) if os.path.exists(os.path.join(VAL_DIR,   cls)) else 0
        grand_train += t
        grand_val   += v
        print(f"  {cls:<15} {t:>7} {v:>7} {t+v:>7}")

    print(f"  {'-'*42}")
    print(f"  {'TOTAL':<15} {grand_train:>7} {grand_val:>7} {grand_train+grand_val:>7}")
    print(f"{'='*60}")
    print(f"\n  Image size : {IMG_SIZE[0]}×{IMG_SIZE[1]} px")
    print(f"  Val split  : {int(VAL_SPLIT*100)}%")
    print(f"\n  Next: python model_training/4_train_model.py")


def main():
    print("=" * 60)
    print("  FORAX — Dataset Preparation")
    print("=" * 60)
    print(f"  Source : {RAW_DIR}")
    print(f"  Output : {TRAIN_DIR}  /  {VAL_DIR}")
    print(f"  Size   : {IMG_SIZE[0]}×{IMG_SIZE[1]}  |  Val split: {int(VAL_SPLIT*100)}%")
    print(f"  Max/class: {MAX_PER_CLASS}  |  Classes: {', '.join(CLASSES)}")
    print(f"  Augment: {AUGMENT_LOW_CLASSES}  |  Target train/class: {TARGET_TRAIN_PER_CLASS}  |  Max aug/src: {MAX_AUG_PER_IMAGE}\n")

    if IMG_SIZE[0] <= 0:
        print("[FAIL] FORAX_IMG_SIZE must be a positive integer")
        sys.exit(1)
    if not (0 < VAL_SPLIT < 1):
        print("[FAIL] FORAX_VAL_SPLIT must be between 0 and 1")
        sys.exit(1)
    if MAX_PER_CLASS <= 0:
        print("[FAIL] FORAX_MAX_PER_CLASS must be a positive integer")
        sys.exit(1)
    if TARGET_TRAIN_PER_CLASS < 0:
        print("[FAIL] FORAX_TARGET_TRAIN_PER_CLASS cannot be negative")
        sys.exit(1)
    if MAX_AUG_PER_IMAGE <= 0:
        print("[FAIL] FORAX_MAX_AUG_PER_IMAGE must be a positive integer")
        sys.exit(1)

    if not os.path.exists(RAW_DIR):
        print("[FAIL] dataset/raw/ not found. Run step 2 first:")
        print("       python model_training/2_download_datasets.py")
        sys.exit(1)

    for cls in CLASSES:
        print(f"\n  Processing: {cls.upper()}")
        train_n, val_n, aug_n = prepare_class(cls)
        if aug_n > 0:
            print(f"  -> train: {train_n}  |  val: {val_n}  |  augmented: {aug_n}")
        else:
            print(f"  -> train: {train_n}  |  val: {val_n}")

    print_dataset_summary()


if __name__ == "__main__":
    main()
