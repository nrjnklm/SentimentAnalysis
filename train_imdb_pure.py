"""
Stage 2: Train pure-Python TF-IDF + Logistic Regression on IMDb dataset.

Uses the existing pure-Python ML pipeline classes from ml_pipeline.py
(TfidfVectorizer, LogisticRegression, LabelEncoder) — no scikit-learn,
no compiled extensions, no Windows Application Control triggers.

Steps:
1. Load dataset/imdb_train.csv and dataset/imdb_test.csv
2. Remove the 123 overlapping reviews from the training set only
3. Careful text preprocessing: strip HTML, lowercase, preserve negation/intensifiers,
   avoid over-aggressive punctuation/digit removal
4. Train TF-IDF + Logistic Regression (pure Python)
5. Evaluate on untouched official IMDb test set
6. Save artifacts to models/ (new filenames, preserving existing files)
7. Verify zero overlap, successful loading, predictions on 5+ diverse reviews
"""

import csv
import json
import math
import os
import re
import string
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")

TRAIN_CSV = os.path.join(DATASET_DIR, "imdb_train.csv")
TEST_CSV = os.path.join(DATASET_DIR, "imdb_test.csv")

# New artifact filenames (preserve existing synthetic-model artifacts)
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model_imdb.json")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer_imdb.json")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_imdb.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_imdb.json")

# Import pure-Python pipeline classes from ml_pipeline.py
sys.path.insert(0, BASE_DIR)
from ml_pipeline import (
    TfidfVectorizer,
    LogisticRegression,
    LabelEncoder,
    clean_text as ml_clean_text,
)


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
    """
    Careful preprocessing for IMDb reviews:
    - Strip HTML tags (e.g. <br />)
    - Lowercase
    - Collapse whitespace
    - Preserve negation words, intensifiers, punctuation, digits
    """
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Lowercase
    text = text.lower()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def save_artifacts(model, vectorizer, label_encoder):
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2)
    with open(VECTORIZER_PATH, "w", encoding="utf-8") as f:
        json.dump(vectorizer.to_dict(), f, indent=2)
    with open(LABEL_ENCODER_PATH, "w", encoding="utf-8") as f:
        json.dump(label_encoder.to_dict(), f, indent=2)
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"   Label encoder saved to: {LABEL_ENCODER_PATH}")


def load_artifacts():
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        model = LogisticRegression.from_dict(json.load(f))
    with open(VECTORIZER_PATH, "r", encoding="utf-8") as f:
        vectorizer = TfidfVectorizer.from_dict(json.load(f))
    with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        label_encoder = LabelEncoder.from_dict(json.load(f))
    return model, vectorizer, label_encoder


def predict_clean(text, model, vectorizer, le):
    cleaned = preprocess_text(text)
    tfidf = vectorizer.transform([cleaned])[0]
    probs = model.predict_proba([tfidf])[0]
    pred_int = model.predict([tfidf])[0]
    label = le.inverse_transform([pred_int])[0]
    confidence = max(probs)
    return label, confidence, probs


def compute_metrics(y_true, y_pred, labels):
    """Compute accuracy, precision, recall, F1, confusion matrix."""
    n = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / n if n > 0 else 0.0

    # Per-class metrics for positive class (label index 1)
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cm = [[tn, fp], [fn, tp]]  # [[true neg, false pos], [false neg, true pos]]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Step 1: Load datasets ---
    print("Loading IMDb datasets...")
    train_texts, train_labels = load_csv(TRAIN_CSV)
    test_texts, test_labels = load_csv(TEST_CSV)
    print(f"  Train (original): {len(train_texts)} reviews")
    print(f"  Test:             {len(test_texts)} reviews")

    # --- Step 2: Remove overlaps from training set only ---
    train_texts, train_labels, removed = remove_overlaps(train_texts, train_labels, test_texts)
    print(f"\nRemoved {removed} overlapping reviews from training set")
    print(f"  Train (after dedup): {len(train_texts)} reviews")
    print(f"  Test (untouched):    {len(test_texts)} reviews")

    # Verify zero overlap
    train_set = set(train_texts)
    test_set = set(test_texts)
    overlap = train_set & test_set
    print(f"  Train/test overlap after dedup: {len(overlap)} (must be 0)")

    # --- Step 3: Preprocess ---
    print("\nPreprocessing text (careful: preserve negation, intensifiers, punctuation)...")
    train_texts_clean = [preprocess_text(t) for t in train_texts]
    test_texts_clean = [preprocess_text(t) for t in test_texts]

    # --- Step 4: Encode labels ---
    le = LabelEncoder()
    le.fit(train_labels)
    y_train = le.transform(train_labels)
    y_test = le.transform(test_labels)
    print(f"\nLabel encoding: {list(le.classes_)} -> {list(le.transform(le.classes_))}")

    # --- Step 5: Train TF-IDF + Logistic Regression ---
    print("\nTraining TF-IDF + Logistic Regression (pure Python)...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(train_texts_clean)
    X_test_tfidf = vectorizer.transform(test_texts_clean)
    print(f"  Training features: {len(X_train_tfidf)} x {vectorizer.n_features_in_}")
    print(f"  Testing features:  {len(X_test_tfidf)} x {vectorizer.n_features_in_}")

    model = LogisticRegression(lr=0.5, n_iter=800, C=1.0, random_state=42)
    model.fit(X_train_tfidf, y_train)
    print("  Training complete.")

    # --- Step 6: Evaluate on test set ---
    print("\nEvaluating on untouched IMDb test set...")
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)

    metrics = compute_metrics(y_test, y_pred, le.classes_)
    cm = metrics["confusion_matrix"]

    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"                    predicted negative  predicted positive")
    print(f"actual negative        {cm[0][0]:>5}               {cm[0][1]:>5}")
    print(f"actual positive        {cm[1][0]:>5}               {cm[1][1]:>5}")

    # --- Step 7: Save artifacts ---
    print("\nSaving artifacts to models/...")
    save_artifacts(model, vectorizer, le)

    metrics_report = {
        "dataset": "IMDb Large Movie Review",
        "train_samples": len(train_texts_clean),
        "test_samples": len(test_texts_clean),
        "overlaps_removed_from_train": removed,
        "train_test_overlap_after_dedup": len(overlap),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "confusion_matrix": cm,
        "label_mapping": {str(i): cls for i, cls in enumerate(le.classes_)},
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"  Metrics saved to: {METRICS_PATH}")

    # --- Step 8: Verify loading and predictions ---
    print("\n=== Verification: Load saved model and predict ===")
    loaded_model, loaded_vectorizer, loaded_le = load_artifacts()

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

    print(f"\n{'Review':<58} {'Predicted':<12} {'Confidence':<12}")
    print("-" * 84)
    for text in sample_reviews:
        label, confidence, probs = predict_clean(text, loaded_model, loaded_vectorizer, loaded_le)
        excerpt = text[:56] + "..." if len(text) > 56 else text
        print(f"{excerpt:<58} {label:<12} {confidence:.4f}")

    print("\nStage 2 complete. Artifacts saved and verified.")
    return metrics_report


if __name__ == "__main__":
    main()
