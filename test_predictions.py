"""
Independent test script for the Sentiment Analysis ML pipeline.

Loads the SAME fitted artifacts the Streamlit app uses
(models/sentiment_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl)
and asserts that the six required reviews are predicted correctly
by the actual model (nothing is hardcoded).
"""
import json
import os
import sys

import joblib

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
METRICS_PATH = "metrics.json"

# (review, expected_sentiment)
TEST_CASES = [
    ("Its not good at all", "negative"),
    ("This product is terrible", "negative"),
    ("I hate this product", "negative"),
    ("This product is excellent", "positive"),
    ("I love this product", "positive"),
    ("Very happy with my purchase", "positive"),
]

PASS = True


def clean_text(text):
    if not isinstance(text, str):
        return ""
    import re, string
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    global PASS
    print("=" * 60)
    print("SENTIMENT ANALYSIS - INDEPENDENT PREDICTION TEST")
    print("=" * 60)

    # ---- Verify artifacts exist ----
    for path in [MODEL_PATH, VECTORIZER_PATH, ENCODER_PATH]:
        if not os.path.exists(path):
            print(f"MISSING artifact: {path}")
            print("Run: py -3.13 train_model.py")
            sys.exit(1)

    # ---- Load artifacts (exactly as the app does) ----
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print(f"Classes (label_encoder): {list(label_encoder.classes_)}")
    print("   (0 = negative, 1 = positive)")

    # ---- Load training metrics ----
    metrics = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print("\nTraining evaluation (test set):")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1-Score : {metrics['f1_score']:.4f}")
    else:
        print("WARNING: metrics.json not found")

    # ---- Run the six required predictions through the real model ----
    print("\nPredictions (model must NOT hardcode these):")
    all_ok = True
    for review, expected in TEST_CASES:
        cleaned = clean_text(review)
        tfidf = vectorizer.transform([cleaned])
        pred_int = model.predict(tfidf)[0]
        proba = model.predict_proba(tfidf)[0]
        confidence = float(max(proba))
        pred_label = label_encoder.inverse_transform([pred_int])[0]
        ok = (pred_label == expected)
        all_ok = all_ok and ok
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {review!r:40} expected={expected:9} "
              f"predicted={pred_label:9} confidence={confidence:.2%}")

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: ALL SIX TEST CASES PASSED")
    else:
        print("RESULT: SOME TEST CASES FAILED")
        PASS = False
    print("=" * 60)

    # ---- Sanity checks on metrics ----
    if metrics:
        assert metrics["accuracy"] >= 0.9, "Accuracy too low"
        assert metrics["precision"] >= 0.9, "Precision too low"
        assert metrics["recall"] >= 0.9, "Recall too low"
        assert metrics["f1_score"] >= 0.9, "F1 too low"
        print("\nMetric thresholds (>= 0.90): PASSED")

    assert all_ok, "Not all test cases passed"
    print("\nALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
