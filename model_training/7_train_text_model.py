"""
FORAX — Step 7: Train Text/NLP Model
Trains a multi-class text classifier for forensic categories.

Usage examples:
  python model_training/7_train_text_model.py \
    --source violence:./text_data/raw/violence/violence.csv:text \
    --source threat:./text_data/raw/threat/threat.csv:comment_text:threat:1 \
    --source harassment:./text_data/raw/harassment/harassment.csv:description

Source format:
  class_name:csv_path:text_column[:label_column[:label_value]]

If label_column is provided, rows are filtered where label_column == label_value
(default label_value is 1).
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
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    print("[FAIL] scikit-learn or joblib not installed. Run: pip install scikit-learn joblib")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "nlp_text_model.joblib")
LABELS_PATH = os.path.join(BASE_DIR, "text_label_map.json")
REPORT_PATH = os.path.join(RESULTS_DIR, "text_classification_report.txt")
CM_PATH = os.path.join(RESULTS_DIR, "text_confusion_matrix.json")

DEFAULT_CLASSES = [
    "normal",
    "violence",
    "threat",
    "drugs",
    "weapon",
    "trafficking",
    "harassment",
    "fraud",
    "scam",
]


def parse_source(src):
    parts = src.split(":")
    if len(parts) < 3:
        raise ValueError("Invalid --source format. Use class:csv_path:text_column[:label_column[:label_value]]")
    cls = parts[0].strip()
    path = parts[1].strip()
    text_col = parts[2].strip()
    label_col = parts[3].strip() if len(parts) >= 4 else None
    label_val = parts[4].strip() if len(parts) >= 5 else None
    return cls, path, text_col, label_col, label_val


def load_sources(sources, min_words, allowed_classes):
    texts = []
    labels = []

    for src in sources:
        cls, path, text_col, label_col, label_val = parse_source(src)
        if allowed_classes and cls not in allowed_classes:
            print(f"[SKIP] Class '{cls}' not in --classes list")
            continue
        if not os.path.exists(path):
            print(f"[WARN] Missing file: {path}")
            continue

        df = pd.read_csv(path)
        if text_col not in df.columns:
            print(f"[WARN] Column '{text_col}' not found in {path}")
            continue

        if label_col:
            if label_col not in df.columns:
                print(f"[WARN] Label column '{label_col}' not found in {path}")
                continue
            if label_val is None:
                label_val = "1"
            df = df[df[label_col].astype(str) == str(label_val)]

        series = df[text_col].dropna().astype(str)
        for text in series:
            if len(text.split()) < min_words:
                continue
            texts.append(text)
            labels.append(cls)

        print(f"[OK] Loaded {len(series)} rows for class '{cls}' from {path}")

    return texts, labels


def save_confusion_matrix(cm, labels):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {"labels": labels, "matrix": cm.tolist()}
    with open(CM_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Confusion matrix saved: {CM_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train FORAX text classifier")
    parser.add_argument("--source", action="append", required=True, help="class:csv_path:text_column[:label_column[:label_value]]")
    parser.add_argument("--test-split", type=float, default=0.2, help="Validation split (default: 0.2)")
    parser.add_argument("--min-words", type=int, default=3, help="Minimum words per text (default: 3)")
    parser.add_argument("--max-features", type=int, default=50000, help="Max TF-IDF features (default: 50000)")
    parser.add_argument("--ngram-max", type=int, default=2, help="Max n-gram size (default: 2)")
    parser.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES), help="Comma-separated class list")
    args = parser.parse_args()

    if not (0 < args.test_split < 1):
        print("[FAIL] --test-split must be between 0 and 1")
        sys.exit(1)
    if args.min_words <= 0:
        print("[FAIL] --min-words must be positive")
        sys.exit(1)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        classes = DEFAULT_CLASSES

    print("=" * 60)
    print("  FORAX — Train Text Model")
    print("=" * 60)
    print(f"  Classes: {', '.join(classes)}")

    texts, labels = load_sources(args.source, args.min_words, classes)
    if not texts:
        print("[FAIL] No training samples loaded")
        sys.exit(1)

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=args.test_split, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(1, args.ngram_max),
            min_df=2,
            max_df=0.9
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating...")
    y_pred = pipeline.predict(X_val)
    report = classification_report(y_val, y_pred, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=sorted(set(labels)))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] Report saved: {REPORT_PATH}")

    save_confusion_matrix(cm, sorted(set(labels)))

    print("\nSaving model...")
    joblib.dump(pipeline, MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({"classes": sorted(set(labels))}, f, indent=2)
    print(f"[OK] Model saved: {MODEL_PATH}")

    print("\nDONE")


if __name__ == "__main__":
    main()
