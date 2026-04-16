"""
FORAX — Step 6: Download Text/Tabular Datasets (Kaggle)
Downloads text datasets used for NLP training. This script only downloads and
extracts data. You still need to map columns when training.

Run: python model_training/6_download_text_datasets.py
"""

import os
import sys
import shutil
import kaggle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_RAW_DIR = os.path.join(BASE_DIR, "text_data", "raw")

DATASETS = [
    {
        "id": "shawhin/fake-job-postings",
        "label": "scam",
        "desc": "Fake job postings (scam / fraud-style text)"
    },
    {
        "id": "azminetoushikwasi/woman-harassment-dataset-200121-bangladesh",
        "label": "harassment",
        "desc": "Harassment records (text fields in CSV)"
    },
    {
        "id": "algosforgood/uk-human-trafficking-data",
        "label": "trafficking",
        "desc": "Human trafficking reports/statistics (text fields in CSV)"
    }
]


def download_dataset(dataset_info):
    ds_id = dataset_info["id"]
    label = dataset_info["label"]
    desc = dataset_info["desc"]
    dest_dir = os.path.join(TEXT_RAW_DIR, label)

    print(f"\n{'-'*60}")
    print(f"  Class   : {label.upper()}")
    print(f"  Dataset : {ds_id}")
    print(f"  Purpose : {desc}")
    print(f"{'-'*60}")

    if os.path.exists(dest_dir) and os.listdir(dest_dir):
        print("  [SKIP] Already downloaded")
        return True

    os.makedirs(dest_dir, exist_ok=True)
    tmp_dir = os.path.join(TEXT_RAW_DIR, f"_tmp_{label}")
    os.makedirs(tmp_dir, exist_ok=True)

    try:
        print("  Downloading from Kaggle...")
        kaggle.api.dataset_download_files(ds_id, path=tmp_dir, unzip=True, quiet=False)

        # Move all files to dest_dir (keep structure flat)
        moved = 0
        for root, _, files in os.walk(tmp_dir):
            for fname in files:
                src = os.path.join(root, fname)
                dst = os.path.join(dest_dir, fname)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(fname)
                    dst = os.path.join(dest_dir, f"{base}_dup{ext}")
                shutil.copy2(src, dst)
                moved += 1

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"  [OK] Saved {moved} files to: {dest_dir}")
        return True

    except Exception as e:
        print(f"  [FAIL] {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n  Manual fix: Download '{ds_id}' from kaggle.com/datasets/{ds_id}")
        print(f"  and place files into: {dest_dir}")
        return False


def main():
    print("=" * 60)
    print("  FORAX — Text Dataset Downloader")
    print("=" * 60)
    print(f"  Saving to: {TEXT_RAW_DIR}")

    os.makedirs(TEXT_RAW_DIR, exist_ok=True)

    try:
        kaggle.api.authenticate()
        print("  [OK] Kaggle API authenticated\n")
    except Exception as e:
        print(f"  [FAIL] Kaggle authentication failed: {e}")
        print("  Run python model_training/1_setup_kaggle.py first")
        sys.exit(1)

    results = {}
    for ds in DATASETS:
        results[ds["label"]] = download_dataset(ds)

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n  [WARN] Failed to download: {', '.join(failed)}")

    print("\nNext: Use the CSV files in text_data/raw/ with 7_train_text_model.py")


if __name__ == "__main__":
    main()
