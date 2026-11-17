"""
Independent test script for the Sentiment Analysis ML pipeline.

Loads the SAME fitted artifacts the Streamlit app uses
(models/sentiment_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl)
and asserts that the six required reviews are predicted correctly
by the actual model (nothing is hardcoded).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_pipeline import load_artifacts, predict_clean

# (review, expected_sentiment)
TEST_CASES = [
    ("very good", "Positive"),
    ("excellent product", "Positive"),
    ("I love this", "Positive"),
    ("very bad", "Negative"),
    ("terrible product", "Negative"),
    ("not good at all", "Negative"),
]

PASS = True


def main():
    global PASS
    print("=" * 70)
    print("SENTIMENT ANALYSIS - INDEPENDENT PREDICTION TEST")
    print("=" * 70)

    # ---- Load artifacts (exactly as the app does) ----
    model, vectorizer, label_encoder = load_artifacts()
    print(f"Classes (label_encoder): {list(label_encoder.classes_)}")
    print("   (0 = negative, 1 = positive)")

    # ---- Run the six required predictions through the real model ----
    print("\nPredictions (model must NOT hardcode these):")
    print(f"{'Review':<40} {'Expected':<12} {'Predicted':<12} {'Confidence':<12} {'Match'}")
    print("-" * 70)

    all_ok = True
    for review, expected in TEST_CASES:
        pred, confidence, probs = predict_clean(review, model, vectorizer, label_encoder)
        pred_capitalized = pred.capitalize()
        ok = (pred_capitalized == expected)
        all_ok = all_ok and ok
        flag = "PASS" if ok else "FAIL"
        print(f"{review:<40} {expected:<12} {pred_capitalized:<12} {confidence:.4f}        {flag}")

    print("-" * 70)
    print("\nDetailed probability breakdown:")
    for review, expected in TEST_CASES:
        pred, confidence, probs = predict_clean(review, model, vectorizer, label_encoder)
        print(f"  {review!r:30} -> P(negative)={probs[0]:.4f}, P(positive)={probs[1]:.4f}, "
              f"predicted={pred} (confidence={confidence:.4f})")

    print("\n" + "=" * 70)
    if all_ok:
        print("RESULT: ALL SIX TEST CASES PASSED")
    else:
        print("RESULT: SOME TEST CASES FAILED")
        PASS = False
    print("=" * 70)

    assert all_ok, "Not all test cases passed"
    print("\nALL ASSERTIONS PASSED.")


if __name__ == "__main__":
    main()
