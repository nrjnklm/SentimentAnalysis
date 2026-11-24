"""
Stage 5.2: Train improved pure-Python TF-IDF + Logistic Regression on IMDb dataset.

Improvements over baseline (train_imdb_pure.py):
1. Punctuation removal in preprocessing (consolidates "brilliant", "brilliant.", "brilliant," into one token)
2. Stochastic Gradient Descent (SGD) for faster convergence
3. Lower learning rate (0.01 vs 0.5)
4. More epochs (50 vs 800 iterations of batch GD)
5. Convergence monitoring (log loss tracking)
6. Early stopping with best-weight restoration
7. Same dedup approach: remove overlaps from training set only

Artifacts saved with _improved suffix (baseline files preserved).
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

# Improved artifact filenames (baseline files preserved)
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model_improved.json")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer_improved.json")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_improved.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics_improved.json")

# Import pure-Python pipeline classes from ml_pipeline.py
sys.path.insert(0, BASE_DIR)
from ml_pipeline import TfidfVectorizer, LabelEncoder, _tokenize


# --- Improved preprocessing -----------------------------------------------------
def preprocess_improved(text):
    """
    Improved preprocessing for IMDb reviews:
    - Strip HTML tags (e.g. <br />)
    - Lowercase
    - Remove punctuation (consolidates "brilliant", "brilliant.", "brilliant,")
    - Collapse whitespace
    - Preserve negation words (contractions like "don't" -> "dont" match NEGATION_WORDS)
    - Preserve digits (useful for neutral/factual review detection)
    """
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Improved Logistic Regression with SGD ------------------------------------
class LogisticRegressionImproved:
    """Binary logistic regression with SGD, L2 regularization,
    convergence monitoring, and early stopping with best-weight restoration."""

    def __init__(self, lr=0.01, n_iter=50, C=1.0, random_state=42,
                 tol=1e-4, patience=5, loss_every=5):
        self.lr = lr
        self.n_iter = n_iter
        self.C = C
        self.random_state = random_state
        self.tol = tol
        self.patience = patience
        self.loss_every = loss_every
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.loss_history_ = []
        self.n_iter_ = 0

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def _compute_loss(self, X, y):
        """Average log loss with clipping to avoid log(0)."""
        n = len(X)
        total_loss = 0.0
        for i in range(n):
            row = X[i]
            z = self.intercept_ + sum(self.coef_[j] * v for j, v in row.items())
            p = self._sigmoid(z)
            p = max(1e-15, min(1.0 - 1e-15, p))
            total_loss += -(y[i] * math.log(p) + (1.0 - y[i]) * math.log(1.0 - p))
        return total_loss / n

    def fit(self, X, y):
        """X: list of sparse feature vectors (dicts); y: list of int labels (0 or 1)."""
        self.n_features_in_ = max(max(row.keys()) for row in X if row) + 1 if any(X) else 0
        n = len(X)
        self.classes_ = [0, 1]

        import random
        rng = random.Random(self.random_state)
        self.coef_ = {j: rng.gauss(0, 0.01) for j in range(self.n_features_in_)}
        self.intercept_ = 0.0

        alpha = 1.0 / (self.C * n)

        best_loss = float("inf")
        best_coef = None
        best_intercept = 0.0
        patience_counter = 0

        for epoch in range(self.n_iter):
            # Shuffle indices for each epoch
            indices = list(range(n))
            rng.shuffle(indices)

            for i in indices:
                row = X[i]
                z = self.intercept_ + sum(self.coef_[j] * v for j, v in row.items())
                p = self._sigmoid(z)
                error = p - y[i]

                # Update intercept
                self.intercept_ -= self.lr * error

                # Update weights for non-zero features only
                for j, v in row.items():
                    grad = error * v + alpha * self.coef_[j]
                    self.coef_[j] -= self.lr * grad

            # Compute loss periodically for convergence monitoring
            if epoch % self.loss_every == 0 or epoch == self.n_iter - 1:
                loss = self._compute_loss(X, y)
                self.loss_history_.append(loss)
                self.n_iter_ = epoch + 1

                if loss < best_loss - self.tol:
                    best_loss = loss
                    best_coef = dict(self.coef_)
                    best_intercept = self.intercept_
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"   Early stopping at epoch {epoch + 1}/{self.n_iter} "
                              f"(best loss: {best_loss:.6f}, current: {loss:.6f})")
                        break

        # Restore best weights
        if best_coef is not None:
            self.coef_ = best_coef
            self.intercept_ = best_intercept

        print(f"   Training stopped after {self.n_iter_} epochs "
              f"(final loss: {self.loss_history_[-1]:.6f}, best loss: {best_loss:.6f})")
        return self

    def predict_proba(self, X):
        """Return list of [P(negative), P(positive)] for each sample."""
        probs = []
        for row in X:
            z = self.intercept_ + sum(self.coef_[j] * v for j, v in row.items())
            p_pos = self._sigmoid(z)
            probs.append([1.0 - p_pos, p_pos])
        return probs

    def predict(self, X):
        """Return list of int labels (0 or 1)."""
        probs = self.predict_proba(X)
        return [1 if p[1] >= 0.5 else 0 for p in probs]

    def to_dict(self):
        return {
            "lr": self.lr,
            "n_iter": self.n_iter,
            "C": self.C,
            "random_state": self.random_state,
            "tol": self.tol,
            "patience": self.patience,
            "loss_every": self.loss_every,
            "coef_": self.coef_,
            "intercept_": self.intercept_,
            "classes_": self.classes_,
            "n_features_in_": self.n_features_in_,
            "loss_history_": self.loss_history_,
            "n_iter_": self.n_iter_,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(lr=d["lr"], n_iter=d["n_iter"], C=d["C"],
                random_state=d["random_state"],
                tol=d.get("tol", 1e-4), patience=d.get("patience", 5),
                loss_every=d.get("loss_every", 5))
        m.coef_ = {int(k): v for k, v in d["coef_"].items()}
        m.intercept_ = d["intercept_"]
        m.classes_ = d["classes_"]
        m.n_features_in_ = d["n_features_in_"]
        m.loss_history_ = d.get("loss_history_", [])
        m.n_iter_ = d.get("n_iter_", d["n_iter"])
        return m


# --- Data loading ---------------------------------------------------------------
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


# --- Artifact save/load ---------------------------------------------------------
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
        model = LogisticRegressionImproved.from_dict(json.load(f))
    with open(VECTORIZER_PATH, "r", encoding="utf-8") as f:
        vectorizer = TfidfVectorizer.from_dict(json.load(f))
    with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        label_encoder = LabelEncoder.from_dict(json.load(f))
    return model, vectorizer, label_encoder


# --- Metrics --------------------------------------------------------------------
def compute_metrics(y_true, y_pred, labels):
    """Compute accuracy, precision, recall, F1, confusion matrix."""
    n = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    accuracy = correct / n if n > 0 else 0.0

    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cm = [[tn, fp], [fn, tp]]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# --- Prediction -----------------------------------------------------------------
def predict_clean_improved(text, model, vectorizer, le):
    """Clean text using improved preprocessing then predict sentiment."""
    cleaned = preprocess_improved(text)
    tfidf = vectorizer.transform([cleaned])[0]
    probs = model.predict_proba([tfidf])[0]
    pred_int = model.predict([tfidf])[0]
    label = le.inverse_transform([pred_int])[0]
    confidence = max(probs)
    return label, confidence, probs


# --- Main training pipeline ------------------------------------------------------
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Step 1: Load datasets ---
    print("Loading IMDb datasets...", flush=True)
    train_texts, train_labels = load_csv(TRAIN_CSV)
    test_texts, test_labels = load_csv(TEST_CSV)
    print(f"  Train (original): {len(train_texts)} reviews", flush=True)
    print(f"  Test:             {len(test_texts)} reviews", flush=True)

    # --- Step 2: Remove overlaps from training set only ---
    train_texts, train_labels, removed = remove_overlaps(train_texts, train_labels, test_texts)
    print(f"\nRemoved {removed} overlapping reviews from training set", flush=True)
    print(f"  Train (after dedup): {len(train_texts)} reviews", flush=True)
    print(f"  Test (untouched):    {len(test_texts)} reviews", flush=True)

    train_set = set(train_texts)
    test_set = set(test_texts)
    overlap = train_set & test_set
    print(f"  Train/test overlap after dedup: {len(overlap)} (must be 0)", flush=True)

    # --- Step 3: Preprocess ---
    print("\nPreprocessing text (improved: punctuation removed, negation preserved)...", flush=True)
    train_texts_clean = [preprocess_improved(t) for t in train_texts]
    test_texts_clean = [preprocess_improved(t) for t in test_texts]

    # --- Step 4: Encode labels ---
    le = LabelEncoder()
    le.fit(train_labels)
    y_train = le.transform(train_labels)
    y_test = le.transform(test_labels)
    print(f"\nLabel encoding: {list(le.classes_)} -> {list(le.transform(le.classes_))}", flush=True)

    # --- Step 5: Train TF-IDF + Logistic Regression ---
    print("\nTraining TF-IDF + Logistic Regression (improved SGD)...", flush=True)
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(train_texts_clean)
    X_test_tfidf = vectorizer.transform(test_texts_clean)
    print(f"  Training features: {len(X_train_tfidf)} x {vectorizer.n_features_in_}", flush=True)
    print(f"  Testing features:  {len(X_test_tfidf)} x {vectorizer.n_features_in_}", flush=True)

    model = LogisticRegressionImproved(lr=0.05, n_iter=30, C=1.0, random_state=42,
                                       tol=1e-4, patience=5, loss_every=5)
    model.fit(X_train_tfidf, y_train)

    # --- Step 6: Evaluate on test set ---
    print("\nEvaluating on untouched IMDb test set...", flush=True)
    y_pred = model.predict(X_test_tfidf)
    y_proba = model.predict_proba(X_test_tfidf)

    metrics = compute_metrics(y_test, y_pred, le.classes_)
    cm = metrics["confusion_matrix"]

    print(f"\n=== Evaluation Results ===", flush=True)
    print(f"Accuracy : {metrics['accuracy']:.4f}", flush=True)
    print(f"Precision: {metrics['precision']:.4f}", flush=True)
    print(f"Recall   : {metrics['recall']:.4f}", flush=True)
    print(f"F1-Score : {metrics['f1_score']:.4f}", flush=True)
    print(f"\nConfusion Matrix (rows=true, cols=pred):", flush=True)
    print(f"                    predicted negative  predicted positive", flush=True)
    print(f"actual negative        {cm[0][0]:>5}               {cm[0][1]:>5}", flush=True)
    print(f"actual positive        {cm[1][0]:>5}               {cm[1][1]:>5}", flush=True)

    # --- Step 7: Save artifacts ---
    print("\nSaving artifacts to models/...", flush=True)
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
        "training_config": {
            "preprocessing": "improved (punctuation removed, HTML stripped, lowercase, whitespace collapsed, digits kept)",
            "vectorizer_max_features": 3000,
            "vectorizer_ngram_range": [1, 2],
            "lr": model.lr,
            "n_iter": model.n_iter,
            "C": model.C,
            "random_state": model.random_state,
            "tol": model.tol,
            "patience": model.patience,
            "loss_every": model.loss_every,
            "n_iter_actual": model.n_iter_,
            "final_loss": model.loss_history_[-1] if model.loss_history_ else None,
            "best_loss": min(model.loss_history_) if model.loss_history_ else None,
            "converged": model.n_iter_ < model.n_iter,
        },
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"  Metrics saved to: {METRICS_PATH}", flush=True)

    # --- Step 8: Verify loading and predictions ---
    print("\n=== Verification: Load saved model and predict ===", flush=True)
    loaded_model, loaded_vectorizer, loaded_le = load_artifacts()

    sample_reviews = [
        ("POSITIVE", "This movie was absolutely fantastic. I loved every moment of it."),
        ("POSITIVE", "An excellent film with brilliant acting and a beautiful story."),
        ("POSITIVE", "Nice movie."),
        ("NEGATIVE", "This movie was terrible, boring, and a complete waste of time."),
        ("NEGATIVE", "I hated every minute of this movie. It was awful."),
        ("NEGATION", "This movie was not good at all."),
        ("NEGATION", "I thought it would be bad, but it was actually amazing."),
        ("MIXED", "The acting was excellent, but the story was boring."),
        ("MIXED", "The visuals were beautiful, but I didn't enjoy the movie."),
        ("NEUTRAL", "The movie was released in 2022 and runs for two hours."),
        ("NEUTRAL", "I watched this movie yesterday."),
    ]

    print(f"\n{'Category':<10} {'Predicted':<12} {'Confidence':<12} {'ProbPos':<10}  Review", flush=True)
    print("-" * 100, flush=True)
    for category, text in sample_reviews:
        label, conf, probs = predict_clean_improved(text, loaded_model, loaded_vectorizer, loaded_le)
        p_pos = probs[1]
        excerpt = text[:55] + "..." if len(text) > 55 else text
        print(f"{category:<10} {label:<12} {conf:<12.4f} {p_pos:<10.4f}  {excerpt}", flush=True)

    print("\nStage 5.2 complete. Candidate artifacts saved and verified.", flush=True)
    return metrics_report


if __name__ == "__main__":
    main()
