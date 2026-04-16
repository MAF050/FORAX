"""
FORAX — Step 2: Download Kaggle Datasets
Downloads 5 forensic-category datasets from Kaggle.
Run: python model_training/2_download_datasets.py

Datasets:
  1. puneet6060/intel-image-classification  → normal
    2. iqmansingh/guns-knives-object-detection → weapons
    3. abdulmananraja/real-life-violence-situations → violence
    4. vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images → drugs (medical proxy)
    5. jjeevanprakash/nsfw-detection          → nsfw
"""

import os
import sys
import shutil
import zipfile
import kaggle

# ── CONFIG ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_DIR     = os.path.join(BASE_DIR, "dataset", "raw")

MAX_RAW_PER_CLASS = int(os.getenv("FORAX_MAX_RAW_PER_CLASS", "5000"))
_nsfw_partial_env = os.getenv("FORAX_NSFW_PARTIAL", os.getenv("FORAX_NSFw_PARTIAL", ""))
NSFW_PARTIAL = _nsfw_partial_env.strip().lower() in {"1", "true", "yes"}

# Each entry: (kaggle_owner/dataset_name, folder_name_after_extract)
DATASETS = [
    {
        "id":    "puneet6060/intel-image-classification",
        "label": "normal",
        "desc":  "Normal everyday images (buildings, nature, streets)"
    },
    {
        "id":    "iqmansingh/guns-knives-object-detection",
        "label": "weapons",
        "desc":  "Guns and knives (weapon imagery)"
    },
    {
        "id":    "abdulmananraja/real-life-violence-situations",
        "label": "violence",
        "desc":  "Violence vs. non-violence (fight/assault imagery)"
    },
    {
        "id":    "vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images",
        "label": "drugs",
        "desc":  "Pills, capsules, tablets (drug paraphernalia proxy)"
    },
    {
        "id":    "jjeevanprakash/nsfw-detection",
        "label": "nsfw",
        "desc":  "NSFW / suspicious content (academic dataset)"
    },
    {
        "id": "thedevastator/phishing-sites-screenshot",
        "label": "scam",
        "desc": "Phishing / Scam website screenshots"
    }
]


def _get_selected_datasets():
    """Optionally filter datasets by FORAX_CLASSES env var."""
    classes_env = os.getenv("FORAX_CLASSES", "normal,weapons,violence,drugs,nsfw,scam")
    if not classes_env:
        return DATASETS

    selected_labels = [c.strip() for c in classes_env.split(",") if c.strip()]
    if not selected_labels:
        return DATASETS

    label_set = {d["label"] for d in DATASETS}
    unknown = [c for c in selected_labels if c not in label_set]
    if unknown:
        print(f"  [WARN] Unknown FORAX_CLASSES ignored: {', '.join(unknown)}")

    return [d for d in DATASETS if d["label"] in selected_labels]


def _download_partial_nsfw_images(ds_id: str, tmp_dir: str, max_images: int) -> int:
    """Download a subset of NSFW-labeled images by fetching individual files."""
    if max_images <= 0:
        return 0

    image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    seen = set()
    if os.path.isdir(tmp_dir):
        for fname in os.listdir(tmp_dir):
            fpath = os.path.join(tmp_dir, fname)
            if not os.path.isfile(fpath):
                continue
            ext = os.path.splitext(fname.lower())[1]
            if ext in image_exts:
                seen.add(fname)

    downloaded = len(seen)
    if downloaded:
        print(f"  [INFO] Resuming partial download (already have {downloaded} files in temp)")
    if downloaded >= max_images:
        return max_images
    page_token = None

    while downloaded < max_images:
        resp = kaggle.api.dataset_list_files(ds_id, page_token=page_token, page_size=200)

        for f in resp.files:
            name = getattr(f, "name", "")
            if "/NSFW/" not in name.replace("\\", "/"):
                continue
            ext = os.path.splitext(name.lower())[1]
            if ext not in image_exts:
                continue

            base = os.path.basename(name)
            if base in seen:
                continue

            try:
                kaggle.api.dataset_download_file(ds_id, name, path=tmp_dir, force=False, quiet=True)
                seen.add(base)
                downloaded += 1
            except Exception:
                continue

            if downloaded >= max_images:
                break

        page_token = getattr(resp, "nextPageToken", None)
        if not page_token:
            break

    return downloaded


def download_dataset(dataset_info):
    """Download and extract a single Kaggle dataset."""
    ds_id    = dataset_info["id"]
    label    = dataset_info["label"]
    desc     = dataset_info["desc"]
    dest_dir = os.path.join(RAW_DIR, label)

    print(f"\n{'-'*60}")
    print(f"  Class   : {label.upper()}")
    print(f"  Dataset : {ds_id}")
    print(f"  Purpose : {desc}")
    print(f"{'-'*60}")

    if os.path.exists(dest_dir) and len(os.listdir(dest_dir)) > 0:
        count = sum(len(files) for _, _, files in os.walk(dest_dir))
        print(f"  [SKIP] Already downloaded ({count} files found)")
        return True

    os.makedirs(dest_dir, exist_ok=True)
    tmp_zip_dir = os.path.join(RAW_DIR, f"_tmp_{label}")
    os.makedirs(tmp_zip_dir, exist_ok=True)

    try:
        print("  Downloading from Kaggle...")
        if label == "nsfw" and NSFW_PARTIAL:
            cap = MAX_RAW_PER_CLASS if MAX_RAW_PER_CLASS > 0 else 5000
            print(f"  [INFO] NSFW partial mode enabled — downloading up to {cap} images...")
            got = _download_partial_nsfw_images(ds_id, tmp_zip_dir, cap)
            if got <= 0:
                raise Exception("NSFW partial download produced no files")
        else:
            kaggle.api.dataset_download_files(ds_id, path=tmp_zip_dir, unzip=True, quiet=False)

        # Find all image files in the extracted folder
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images_found = []
        for root, dirs, files in os.walk(tmp_zip_dir):
            for fname in files:
                if os.path.splitext(fname.lower())[1] in image_exts:
                    images_found.append(os.path.join(root, fname))

        if not images_found:
            print(f"  [WARN] No images found in downloaded data.")
            return False

        # Move images into dest_dir (flat structure)
        cap = MAX_RAW_PER_CLASS if MAX_RAW_PER_CLASS > 0 else 5000
        print(f"  Moving {min(len(images_found), cap)} of {len(images_found)} images to {label}/ ...")
        for i, img_path in enumerate(images_found[:cap]):
            ext = os.path.splitext(img_path)[1]
            dest_name = f"{label}_{i:05d}{ext}"
            shutil.copy2(img_path, os.path.join(dest_dir, dest_name))

        # Cleanup temp dir
        shutil.rmtree(tmp_zip_dir, ignore_errors=True)

        final_count = len(os.listdir(dest_dir))
        print(f"  [OK] {final_count} images saved to dataset/raw/{label}/")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        shutil.rmtree(tmp_zip_dir, ignore_errors=True)
        print(f"\n  Manual fix: Download '{ds_id}' from kaggle.com/datasets/{ds_id}")
        print(f"  and place images into: {dest_dir}")
        return False


def print_summary(datasets):
    """Print a summary of downloaded data."""
    print(f"\n{'='*60}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    total = 0
    for ds in datasets:
        label   = ds["label"]
        d       = os.path.join(RAW_DIR, label)
        count   = len([f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]) if os.path.exists(d) else 0
        status  = "[OK]  " if count > 0 else "[MISS]"
        total  += count
        print(f"  {status} {label:<12} {count:>5} images")
    print(f"{'-'*60}")
    print(f"  TOTAL              {total:>5} images")
    print(f"{'='*60}")
    if total > 0:
        print(f"\n  Next: python model_training/3_prepare_dataset.py")


def main():
    print("=" * 60)
    print("  FORAX — Kaggle Dataset Downloader")
    print("=" * 60)
    print(f"  Saving to: {RAW_DIR}")

    os.makedirs(RAW_DIR, exist_ok=True)

    # Authenticate
    try:
        kaggle.api.authenticate()
        print("  [OK] Kaggle API authenticated\n")
    except Exception as e:
        print(f"  [FAIL] Kaggle authentication failed: {e}")
        print("  Run python model_training/1_setup_kaggle.py first")
        sys.exit(1)

    datasets = _get_selected_datasets()
    if len(datasets) != len(DATASETS):
        labels = ", ".join(d["label"] for d in datasets)
        print(f"  Downloading only selected classes: {labels}\n")

    results = {}
    for ds in datasets:
        results[ds["label"]] = download_dataset(ds)

    print_summary(datasets)

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n  [WARN] Failed to download: {', '.join(failed)}")
        print("  You can add images manually to dataset/raw/<class_name>/")
        print("  Then continue with: python model_training/3_prepare_dataset.py")


if __name__ == "__main__":
    main()
