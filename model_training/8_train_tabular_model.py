"""
FORAX — Step 8: Train Tabular Model (Fraud/Scam)
Trains a classifier on numeric/tabular CSV data.

Usage:
  python model_training/8_train_tabular_model.py \
    --csv D:/path/to/creditcard.csv --label-col Class
"""

import argparse
import json
import os
import sys

try:
    import pandas as pd
except ImportError:
    print("[FAIL] pandas not installed. Run: pip install pandas")
    sys.exit(1)

try:
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.ensemble import RandomForestClassifier
except ImportError:
    print("[FAIL] scikit-learn or joblib not installed. Run: pip install scikit-learn joblib")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "tabular_model.joblib")
REPORT_PATH = os.path.join(RESULTS_DIR, "tabular_classification_report.txt")
META_PATH = os.path.join(BASE_DIR, "tabular_model_meta.json")


def main():
    parser = argparse.ArgumentParser(description="Train FORAX tabular classifier")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--label-col", required=True, help="Label column name")
    parser.add_argument("--drop-cols", default="", help="Comma-separated columns to drop")
    parser.add_argument("--test-split", type=float, default=0.2, help="Validation split (default: 0.2)")
    parser.add_argument("--n-estimators", type=int, default=200, help="RandomForest trees (default: 200)")
    args = parser.parse_args()

    if not (0 < args.test_split < 1):
        print("[FAIL] --test-split must be between 0 and 1")
        sys.exit(1)

    if not os.path.exists(args.csv):
        print(f"[FAIL] CSV not found: {args.csv}")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    if args.label_col not in df.columns:
        print(f"[FAIL] Label column '{args.label_col}' not found")
        sys.exit(1)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    drop_cols = [c for c in drop_cols if c in df.columns]

    y = df[args.label_col]
    X = df.drop(columns=[args.label_col] + drop_cols)

    # Convert categorical columns to one-hot
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_split, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    print("Training model...")
    clf.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, zero_division=0)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    joblib.dump(clf, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"features": list(X.columns)}, f, indent=2)

    print(f"[OK] Report saved: {REPORT_PATH}")
    print(f"[OK] Model saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
