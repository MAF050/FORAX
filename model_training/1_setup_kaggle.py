"""
FORAX — Step 1: Kaggle Setup Verification
Checks that your Kaggle API key is installed correctly.
Run: python model_training/1_setup_kaggle.py
"""

import os
import sys
import json

KAGGLE_JSON_PATH = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")


def check_kaggle_json():
    print("=" * 60)
    print("  FORAX — Kaggle API Setup Check")
    print("=" * 60)

    # ── Check file exists ──────────────────────────────────────────
    if not os.path.exists(KAGGLE_JSON_PATH):
        print(f"\n[FAIL] kaggle.json not found at: {KAGGLE_JSON_PATH}")
        print("\nFix:")
        print("  1. Go to https://www.kaggle.com → Profile → Settings → API")
        print("  2. Click 'Create New Token' → downloads kaggle.json")
        print(f"  3. Move it to: {KAGGLE_JSON_PATH}")
        print("     (Create the .kaggle folder if it doesn't exist)")
        return False

    # ── Validate JSON ──────────────────────────────────────────────
    try:
        with open(KAGGLE_JSON_PATH, 'r') as f:
            creds = json.load(f)
        if 'username' not in creds or 'key' not in creds:
            print("[FAIL] kaggle.json is malformed — missing username or key.")
            return False
        print(f"\n[OK] kaggle.json found")
        print(f"     Username: {creds['username']}")
        print(f"     Key:      {'*' * 20}{creds['key'][-4:]}")
    except Exception as e:
        print(f"[FAIL] Could not read kaggle.json: {e}")
        return False

    # ── Check kaggle package installed ────────────────────────────
    try:
        import kaggle
        print("[OK] kaggle Python package installed")
    except ImportError:
        print("[FAIL] kaggle package not installed.")
        print("  Run: pip install kaggle")
        return False

    # ── Test API connection ───────────────────────────────────────
    print("\nTesting Kaggle API connection...")
    try:
        api = kaggle.api
        api.authenticate()
        print("[OK] Kaggle API authenticated successfully!")
    except Exception as e:
        print(f"[FAIL] API authentication failed: {e}")
        print("\nMake sure your kaggle.json has the correct credentials.")
        return False

    # ── Check TensorFlow ──────────────────────────────────────────
    print("\nChecking TensorFlow...")
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow {tf.__version__} installed")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] GPU detected: {gpus[0].name} — Training will be FAST!")
        else:
            print("[INFO] No GPU detected — Training on CPU (~30–60 min)")
    except ImportError:
        print("[WARN] TensorFlow not installed. Run: pip install tensorflow")

    # ── Check other packages ──────────────────────────────────────
    packages = {
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'joblib': 'joblib'
    }
    print("\nChecking ML dependencies...")
    all_ok = True
    for imp, pkg in packages.items():
        try:
            __import__(imp)
            print(f"[OK] {pkg}")
        except ImportError:
            print(f"[MISS] {pkg}  →  pip install {pkg}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("  ALL CHECKS PASSED — Ready to download datasets!")
        print("  Next: python model_training/2_download_datasets.py")
    else:
        print("  Install missing packages then re-run this script.")
    print("=" * 60)
    return all_ok


if __name__ == "__main__":
    ok = check_kaggle_json()
    sys.exit(0 if ok else 1)
