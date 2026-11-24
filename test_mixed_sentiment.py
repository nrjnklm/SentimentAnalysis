"""
Test script for mixed sentiment detection (Stage 5.4).

Tests 15 examples across 4 categories:
- Positive (4)
- Negative (3)
- Mixed (5)
- Neutral (3)
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, BASE_DIR)

from mixed_sentiment import load_model, detect_mixed


def main():
    model, vectorizer, le = load_model()

    test_cases = [
        # Positive
        ("Positive", "This movie was absolutely fantastic. I loved every moment of it."),
        ("Positive", "The movie was good."),
        ("Positive", "Nice movie."),
        ("Positive", "The acting and story were both excellent."),
        # Negative
        ("Negative", "This movie was terrible, boring, and a complete waste of time."),
        ("Negative", "I hated every minute of this movie."),
        ("Negative", "The movie was not good at all."),
        # Mixed
        ("Mixed", "The acting was excellent, but the story was boring."),
        ("Mixed", "The visuals were beautiful, but I didn't enjoy the movie."),
        ("Mixed", "The movie started well, but the ending was disappointing."),
        ("Mixed", "The story was boring, although the acting was excellent."),
        ("Mixed", "It had great music but terrible acting."),
        # Neutral
        ("Neutral", "The movie was released in 2022 and runs for two hours."),
        ("Neutral", "I watched this movie yesterday."),
        ("Neutral", "The movie is about a family travelling to another country."),
    ]

    print("=" * 120)
    print("MIXED SENTIMENT DETECTION — STAGE 5.4 EXPERIMENT")
    print("=" * 120)

    correct_count = 0
    total = len(test_cases)

    for idx, (expected, text) in enumerate(test_cases, 1):
        result = detect_mixed(text, model, vectorizer, le)

        # Determine correctness
        if expected == "Mixed":
            correct = result["is_mixed"]
        else:
            correct = (result["final_label"] == expected)

        if correct:
            correct_count += 1

        status = "OK" if correct else "FAIL"

        print(f"\n{'-'*120}")
        print(f"#{idx} [{status}] Expected: {expected}")
        print(f"  Review: \"{text}\"")
        print(f"  Overall: {result['overall_label']} (conf={result['overall_confidence']:.4f}, "
              f"probs=[{result['overall_probs'][0]:.4f}, {result['overall_probs'][1]:.4f}])")
        print(f"  Clauses ({len(result['clauses'])}):")
        for c in result["clauses"]:
            print(f"    - \"{c['text']}\" -> {c['label']} (conf={c['confidence']:.4f})")
        print(f"  Has positive: {result['has_positive']}, Has negative: {result['has_negative']}")
        print(f"  Is mixed: {result['is_mixed']}")
        print(f"  Final: {result['final_label']} (conf={result['final_confidence']:.4f})")
        print(f"  Explanation: {result['explanation']}")

    print(f"\n{'='*120}")
    print(f"ACCURACY: {correct_count}/{total} = {100*correct_count/total:.1f}%")
    print(f"{'='*120}")

    # Save metrics
    metrics = {
        "experiment": "Stage 5.4 — Mixed Sentiment Detection",
        "total_tests": total,
        "correct": correct_count,
        "accuracy": correct_count / total,
        "thresholds": {
            "OVERALL_MIXED_MARGIN": 0.15,
            "CLAUSE_CONF_THRESHOLD": 0.60,
            "MIN_CLAUSE_WORDS": 3,
            "MIN_OPPOSING_CLAUSES": 1,
        },
        "test_results": [
            {
                "id": idx,
                "expected": expected,
                "text": text,
                "overall_label": detect_mixed(text, model, vectorizer, le)["overall_label"],
                "overall_confidence": detect_mixed(text, model, vectorizer, le)["overall_confidence"],
                "num_clauses": len(detect_mixed(text, model, vectorizer, le)["clauses"]),
                "is_mixed": detect_mixed(text, model, vectorizer, le)["is_mixed"],
                "final_label": detect_mixed(text, model, vectorizer, le)["final_label"],
                "final_confidence": detect_mixed(text, model, vectorizer, le)["final_confidence"],
                "correct": (expected == "Mixed" and detect_mixed(text, model, vectorizer, le)["is_mixed"]) or
                           (expected != "Mixed" and detect_mixed(text, model, vectorizer, le)["final_label"] == expected),
            }
            for idx, (expected, text) in enumerate(test_cases, 1)
        ],
    }

    metrics_path = os.path.join(MODELS_DIR, "metrics_mixed_experiment.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
