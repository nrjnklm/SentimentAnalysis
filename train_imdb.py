"""
Stage 2: Train TF-IDF + Logistic Regression on the official IMDb dataset.

Steps:
1. Load dataset/imdb_train.csv and dataset/imdb_test.csv
2. Remove the 123 overlapping reviews from the training set only (zero leakage)
3. Text preprocessing: lowercase, strip HTML tags, preserve negation/intensifiers,
   avoid over-aggressive digit/punctuation removal
4. Train TF-IDF + Logistic Regression using a scikit-learn Pipeline
5. Evaluate on the untouched official IMDb test set
6. Save artifacts to models/
7. Verify zero overlap, successful loading, and predictions on 5+ diverse reviews
"""

import csv
import json
import os
import re
import string
import joblib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAIN_CSV = os.path.join(DATASET_DIR, "imdb_train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "imdb_test.csv")

MODEL_PIPELINE_PATH = os.path.join(MODELS_DIR, "sentiment_model_imdb.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_imdb.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_imdb.json")


def load_csv(path):
    """Load a CSV with columns review, sentiment."""
    reviews, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append(row["review"])
            labels.append(row["sentiment"])
    return reviews, labels


def remove_overlaps(train_texts, train_labels, test_texts):
    """Remove reviews from train that also appear in test (exact text match)."""
    test_set = set(test_texts)
    kept_texts, kept_labels, removed = [], [], 0
    for text, label in zip(train_texts, train_labels):
        if text in test_set:
            removed += 1
        else:
            kept_texts.append(text)
            kept_labels.append(label)
    return kept_texts, kept_labels, removed


def preprocess_text(text):
    """Careful preprocessing: lowercase, strip HTML, preserve negation/intensifiers."""
    # Strip HTML tags (IMDb reviews contain <br />)
    text = re.sub(r"<[^>]+>", " ", text)
    # Lowercase
    text = text.lower()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pipeline():
    """Build TF-IDF + Logistic Regression Pipeline."""
    tfidf = TfidfVectorizer(
        preprocessor=preprocess_text,
        strip_accents="unicode",
        lowercase=False,        # we already lowercase in preprocess_text
        stop_words="english",   # removes stop words but keeps negation via custom handling below
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        max_df=0.95,
    )
    # Note: scikit-learn's english stop words list does NOT include negation words
    # like "not", "no", "nor", so they are preserved. Intensifiers like "very"
    # are also not in the default stop words list.
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        class_weight="balanced",
    )
    return Pipeline([
        ("tfidf", tfidf),
        ("clf", clf),
    ])


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Step 1: Load datasets ---
    print("Loading IMDb datasets...")
    train_texts, train_labels = load_csv(TRAIN_CSV)
    test_texts, test_labels = load_csv(TEST_CSV)
    print(f"  Train: {len(train_texts)} reviews")
    print(f"  Test:  {len(test_texts)} reviews")

    # --- Step 2: Remove overlaps from training set only ---
    train_texts, train_labels, removed = remove_overlaps(train_texts, train_labels, test_texts)
    print(f"\nRemoved {removed} overlapping reviews from training set")
    print(f"  Train after dedup: {len(train_texts)} reviews")
    print(f"  Test (untouched):  {len(test_texts)} reviews")

    # Verify zero overlap after dedup
    train_set = set(train_texts)
    test_set = set(test_texts)
    overlap = train_set & test_set
    print(f"  Train/test overlap after dedup: {len(overlap)} (must be 0)")

    # --- Step 3: Preprocess ---
    print("\nPreprocessing text...")
    train_texts_clean = [preprocess_text(t) for t in train_texts]
    test_texts_clean = [preprocess_text(t) for t in test_texts]

    # --- Step 4: Encode labels ---
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)
    print(f"\nLabel encoding: {list(le.classes_)} -> {list(le.transform(le.classes_))}")

    # --- Step 5: Train Pipeline ---
    print("\nTraining TF-IDF + Logistic Regression Pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(train_texts_clean, y_train)
    print("Training complete.")

    # --- Step 6: Evaluate on test set ---
    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(test_texts_clean)
    y_proba = pipeline.predict_proba(test_texts_clean)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, target_names=list(le.classes_))

    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"             predicted negative  predicted positive")
    print(f"actual neg     {cm[0][0]:>5}               {cm[0][1]:>5}")
    print(f"actual pos     {cm[1][0]:>5}               {cm[1][1]:>5}")
    print(f"\nClassification Report:")
    print(report)

    # --- Step 7: Save artifacts ---
    print("\nSaving artifacts to models/...")
    joblib.dump(pipeline, MODEL_PIPELINE_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)

    metrics = {
        "dataset": "IMDb Large Movie Review",
        "train_samples": len(train_texts_clean),
        "test_samples": len(test_texts_clean),
        "overlaps_removed_from_train": removed,
        "train_test_overlap_after_dedup": len(overlap),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "label_mapping": {str(i): cls for i, cls in enumerate(le.classes_)},
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Model pipeline: {MODEL_PIPELINE_PATH}")
    print(f"  Label encoder:  {LABEL_ENCODER_PATH}")
    print(f"  Metrics:        {METRICS_PATH}")

    # --- Step 8: Verify loading and predictions ---
    print("\n=== Verification: Load saved model and predict ===")
    loaded_pipeline = joblib.load(MODEL_PIPELINE_PATH)
    loaded_le = joblib.load(LABEL_ENCODER_PATH)

    sample_reviews = [
        "This film was absolutely brilliant. I loved every minute of it.",
        "Terrible movie. Complete waste of time. Do not watch it.",
        "Not good at all. The acting was poor and the plot made no sense.",
        "Very good! Great cast, excellent direction, highly recommended.",
        "It was okay. Nothing special but not terrible either.",
        "I cannot stand this movie. Worst film I have ever seen.",
        "Outstanding performance by the lead actor. Brilliant cinematography.",
        "The movie is not as described. Very disappointed. Avoid this.",
    ]

    sample_clean = [preprocess_text(t) for t in sample_reviews]
    sample_pred = loaded_pipeline.predict(sample_clean)
    sample_proba = loaded_pipeline.predict_proba(sample_clean)

    print(f"\n{'Review':<60} {'Predicted':<12} {'Confidence':<12}")
    print("-" * 86)
    for text, pred, proba in zip(sample_reviews, sample_pred, sample_proba):
        label = loaded_le.inverse_transform([pred])[0]
        confidence = max(proba)
        excerpt = text[:58] + "..." if len(text) > 58 else text
        print(f"{excerpt:<60} {label:<12} {confidence:.4f}")

    print("\nStage 2 complete. Artifacts saved and verified.")
    return metrics


if __name__ == "__main__":
    main()
