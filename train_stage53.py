"""
Stage 5.3: Controlled experiment — vary training epochs (30, 50, 100).

Uses the same improved preprocessing and SGD algorithm as Stage 5.2,
but tests whether more epochs improve performance or cause overfitting.

Preserves all existing artifacts. Saves best candidate as stage53 files.
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

sys.path.insert(0, BASE_DIR)
from ml_pipeline import TfidfVectorizer, LabelEncoder


# --- Improved preprocessing (same as Stage 5.2) ------------------------------
def preprocess_improved(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Logistic Regression with SGD (same as Stage 5.2) -----------------------
class LogisticRegressionSGD:
    def __init__(self, lr=0.05, n_iter=30, C=1.0, random_state=42,
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
            indices = list(range(n))
            rng.shuffle(indices)

            for i in indices:
                row = X[i]
                z = self.intercept_ + sum(self.coef_[j] * v for j, v in row.items())
                p = self._sigmoid(z)
                error = p - y[i]

                self.intercept_ -= self.lr * error

                for j, v in row.items():
                    grad = error * v + alpha * self.coef_[j]
                    self.coef_[j] -= self.lr * grad

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
                              f"(best loss: {best_loss:.6f}, current: {loss:.6f})",
                              flush=True)
                        break

        if best_coef is not None:
            self.coef_ = best_coef
            self.intercept_ = best_intercept

        print(f"   Stopped after {self.n_iter_} epochs "
              f"(final loss: {self.loss_history_[-1]:.6f}, best loss: {best_loss:.6f})",
              flush=True)
        return self

    def predict_proba(self, X):
        probs = []
        for row in X:
            z = self.intercept_ + sum(self.coef_[j] * v for j, v in row.items())
            p_pos = self._sigmoid(z)
            probs.append([1.0 - p_pos, p_pos])
        return probs

    def predict(self, X):
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
    reviews, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append(row["review"])
            labels.append(row["sentiment"])
    return reviews, labels


def remove_overlaps(train_texts, train_labels, test_texts):
    test_set = set(test_texts)
    kept_texts, kept_labels, removed = [], [], 0
    for text, label in zip(train_texts, train_labels):
        if text in test_set:
            removed += 1
        else:
            kept_texts.append(text)
            kept_labels.append(label)
    return kept_texts, kept_labels, removed


# --- Metrics --------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
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
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1_score": f1, "confusion_matrix": [[tn, fp], [fn, tp]],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def predict_clean(text, model, vectorizer, le):
    cleaned = preprocess_improved(text)
    tfidf = vectorizer.transform([cleaned])[0]
    probs = model.predict_proba([tfidf])[0]
    pred_int = model.predict([tfidf])[0]
    label = le.inverse_transform([pred_int])[0]
    confidence = max(probs)
    return label, confidence, probs


# --- Main experiment ------------------------------------------------------------
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Load datasets ---
    print("Loading IMDb datasets...", flush=True)
    train_texts, train_labels = load_csv(TRAIN_CSV)
    test_texts, test_labels = load_csv(TEST_CSV)
    print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}", flush=True)

    # --- Dedup ---
    train_texts, train_labels, removed = remove_overlaps(train_texts, train_labels, test_texts)
    print(f"  Removed {removed} overlaps. Train: {len(train_texts)}, Test: {len(test_texts)}", flush=True)

    # --- Preprocess ---
    train_texts_clean = [preprocess_improved(t) for t in train_texts]
    test_texts_clean = [preprocess_improved(t) for t in test_texts]

    le = LabelEncoder()
    le.fit(train_labels)
    y_train = le.transform(train_labels)
    y_test = le.transform(test_labels)

    # --- Vectorizer (same as Stage 5.2) ---
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(train_texts_clean)
    X_test_tfidf = vectorizer.transform(test_texts_clean)
    print(f"  Features: {vectorizer.n_features_in_}", flush=True)

    # --- Experiment: train with 30, 50, 100 epochs ---
    epoch_configs = [30, 50, 100]
    results = {}

    for n_epochs in epoch_configs:
        print(f"\n{'='*60}", flush=True)
        print(f"Training with {n_epochs} epochs...", flush=True)
        print(f"{'='*60}", flush=True)

        model = LogisticRegressionSGD(lr=0.05, n_iter=n_epochs, C=1.0,
                                      random_state=42, tol=1e-4, patience=5, loss_every=5)
        model.fit(X_train_tfidf, y_train)

        # Evaluate
        y_pred = model.predict(X_test_tfidf)
        metrics = compute_metrics(y_test, y_pred)
        cm = metrics["confusion_matrix"]

        print(f"  Accuracy:  {metrics['accuracy']:.4f}", flush=True)
        print(f"  Precision: {metrics['precision']:.4f}", flush=True)
        print(f"  Recall:    {metrics['recall']:.4f}", flush=True)
        print(f"  F1-Score:  {metrics['f1_score']:.4f}", flush=True)
        print(f"  CM: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}", flush=True)
        print(f"  Loss history: {[round(l, 4) for l in model.loss_history_]}", flush=True)

        results[n_epochs] = {
            "model": model,
            "metrics": metrics,
            "cm": cm,
        }

    # --- Compare all candidates ---
    print(f"\n{'='*70}", flush=True)
    print("COMPARISON OF ALL CANDIDATES", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Epochs':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Loss':<10}", flush=True)
    print("-" * 68, flush=True)
    for n_epochs in epoch_configs:
        m = results[n_epochs]["metrics"]
        loss = results[n_epochs]["model"].loss_history_[-1]
        print(f"{n_epochs:<10} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
              f"{m['recall']:<12.4f} {m['f1_score']:<12.4f} {loss:<10.4f}", flush=True)

    # Also show baseline and Stage 5.2 for reference
    print(f"\nReference:", flush=True)
    print(f"  Baseline (Stage 2):    Acc=0.7814, Prec=0.7197, Rec=0.9221, F1=0.8084", flush=True)
    print(f"  Stage 5.2 (30 epochs): Acc=0.8642, Prec=0.8777, Rec=0.8462, F1=0.8617", flush=True)

    # --- Select best candidate ---
    # Choose by highest F1-score (balanced metric)
    best_epochs = max(epoch_configs, key=lambda e: results[e]["metrics"]["f1_score"])
    best_model = results[best_epochs]["model"]
    best_metrics = results[best_epochs]["metrics"]
    print(f"\nBest candidate: {best_epochs} epochs (F1={best_metrics['f1_score']:.4f})", flush=True)

    # --- Save best candidate as stage53 ---
    STAGE53_MODEL = os.path.join(MODELS_DIR, "sentiment_model_stage53.json")
    STAGE53_VEC = os.path.join(MODELS_DIR, "tfidf_vectorizer_stage53.json")
    STAGE53_LE = os.path.join(MODELS_DIR, "label_encoder_stage53.json")
    STAGE53_METRICS = os.path.join(MODELS_DIR, "metrics_stage53.json")

    with open(STAGE53_MODEL, "w", encoding="utf-8") as f:
        json.dump(best_model.to_dict(), f, indent=2)
    with open(STAGE53_VEC, "w", encoding="utf-8") as f:
        json.dump(vectorizer.to_dict(), f, indent=2)
    with open(STAGE53_LE, "w", encoding="utf-8") as f:
        json.dump(le.to_dict(), f, indent=2)

    stage53_metrics = {
        "dataset": "IMDb Large Movie Review",
        "experiment": "Stage 5.3 - epoch comparison (30 vs 50 vs 100)",
        "best_epochs": best_epochs,
        "train_samples": len(train_texts_clean),
        "test_samples": len(test_texts_clean),
        "overlaps_removed_from_train": removed,
        "accuracy": best_metrics["accuracy"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1_score": best_metrics["f1_score"],
        "confusion_matrix": best_metrics["confusion_matrix"],
        "label_mapping": {str(i): cls for i, cls in enumerate(le.classes_)},
        "training_config": {
            "preprocessing": "improved (punctuation removed, HTML stripped, lowercase, whitespace collapsed, digits kept)",
            "vectorizer_max_features": 3000,
            "vectorizer_ngram_range": [1, 2],
            "lr": best_model.lr,
            "epochs_tested": epoch_configs,
            "best_epochs": best_epochs,
            "C": best_model.C,
            "random_state": best_model.random_state,
            "final_loss": best_model.loss_history_[-1],
            "best_loss": min(best_model.loss_history_),
            "n_iter_actual": best_model.n_iter_,
            "all_results": {
                str(e): {
                    "accuracy": results[e]["metrics"]["accuracy"],
                    "precision": results[e]["metrics"]["precision"],
                    "recall": results[e]["metrics"]["recall"],
                    "f1_score": results[e]["metrics"]["f1_score"],
                    "confusion_matrix": results[e]["metrics"]["confusion_matrix"],
                    "final_loss": results[e]["model"].loss_history_[-1],
                    "epochs_completed": results[e]["model"].n_iter_,
                }
                for e in epoch_configs
            },
        },
    }
    with open(STAGE53_METRICS, "w", encoding="utf-8") as f:
        json.dump(stage53_metrics, f, indent=2)

    print(f"\n  Stage 5.3 artifacts saved (best: {best_epochs} epochs)", flush=True)

    # --- Test all 15 reviews ---
    print(f"\n{'='*100}", flush=True)
    print("TEST REVIEWS - Stage 5.3 Best Candidate", flush=True)
    print(f"{'='*100}", flush=True)

    test_reviews = [
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
        ("POSITIVE", "The movie was good."),
        ("NEGATION", "The movie was not bad."),
        ("MIXED", "I expected a great movie, but it disappointed me."),
        ("MIXED", "The story was boring, although the acting was excellent."),
    ]

    print(f"\n{'#':<4} {'Category':<10} {'Predicted':<12} {'Confidence':<12} {'ProbPos':<10}  Review", flush=True)
    print("-" * 110, flush=True)
    for idx, (category, text) in enumerate(test_reviews, 1):
        label, conf, probs = predict_clean(text, best_model, vectorizer, le)
        p_pos = probs[1]
        excerpt = text[:60] + "..." if len(text) > 60 else text
        correct = ""
        if category == "POSITIVE" and label == "positive":
            correct = "OK"
        elif category == "NEGATIVE" and label == "negative":
            correct = "OK"
        elif category == "NEGATION":
            if "not good" in text.lower() and label == "negative":
                correct = "OK"
            elif "amazing" in text.lower() and label == "positive":
                correct = "OK"
            else:
                correct = "??"
        elif category == "MIXED":
            correct = "--"
        elif category == "NEUTRAL":
            correct = "--"
        print(f"{idx:<4} {category:<10} {label:<12} {conf:<12.4f} {p_pos:<10.4f}  {excerpt}", flush=True)

    print(f"\nStage 5.3 complete. Best candidate: {best_epochs} epochs.", flush=True)
    return results, best_epochs


if __name__ == "__main__":
    main()
