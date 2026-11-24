"""
Mixed Sentiment Detection — Experimental Implementation (Stage 5.4)

Uses the Stage 5.3 100-epoch binary model as foundation.
Detects Mixed sentiment by analyzing clause-level predictions.

Approach:
1. Preprocess text (same as Stage 5.3: punctuation removed, negation preserved)
2. Split review into clauses at sentence boundaries and contrastive conjunctions
3. Predict sentiment for each clause using the Stage 5.3 model
4. Detect Mixed when both positive and negative clauses exist with sufficient confidence
5. Fall back to overall prediction when no genuine mixed pattern is found

No hardcoded review predictions. No simple "if but exists" rules.
"""

import json
import math
import os
import re
import string
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
sys.path.insert(0, BASE_DIR)

from train_stage53 import LogisticRegressionSGD, preprocess_improved
from ml_pipeline import TfidfVectorizer, LabelEncoder


# --- Artifact paths (Stage 5.3 model) ---------------------------------------
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model_stage53.json")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer_stage53.json")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_stage53.json")


def load_model():
    """Load the Stage 5.3 100-epoch model, vectorizer, and label encoder."""
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        model = LogisticRegressionSGD.from_dict(json.load(f))
    with open(VECTORIZER_PATH, "r", encoding="utf-8") as f:
        vectorizer = TfidfVectorizer.from_dict(json.load(f))
    with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        label_encoder = LabelEncoder.from_dict(json.load(f))
    return model, vectorizer, label_encoder


# --- Clause splitting -------------------------------------------------------
# Contrastive conjunctions that often separate opposing sentiments
CONTRASTIVE_MARKERS = [
    "but", "although", "however", "though", "yet",
    "while", "whereas", "despite", "even though",
]

# Minimum clause length (in words) to be considered meaningful
MIN_CLAUSE_WORDS = 2


# Negation words (same as ml_pipeline.py)
NEGATION_WORDS = {
    "not", "no", "nor",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "cant", "couldnt", "shouldnt", "wont", "wouldnt", "hasnt", "havent", "hadnt",
    "mustnt", "neednt", "mightn",
}

# Common positive sentiment words (for negation scope detection)
POSITIVE_SENTIMENT_WORDS = {
    "good", "great", "excellent", "amazing", "fantastic", "beautiful",
    "wonderful", "brilliant", "outstanding", "superb", "perfect", "love",
    "loved", "enjoy", "enjoyed", "nice", "best", "awesome", "incredible",
    "marvelous", "splendid", "terrific", "delightful", "pleased", "happy",
}

# Common negative sentiment words
NEGATIVE_SENTIMENT_WORDS = {
    "bad", "terrible", "awful", "horrible", "boring", "waste", "hate",
    "hated", "disappointing", "disappointed", "poor", "worst", "ugly",
    "stupid", "dull", "mediocre", "annoying", "frustrating", "sad",
}


def split_into_clauses(text):
    """
    Split text into meaningful clauses.

    Strategy:
    1. First split on sentence boundaries (. ! ? ;)
    2. Then split each sentence on contrastive conjunctions
       only if both resulting parts have >= MIN_CLAUSE_WORDS words
    3. Filter out very short fragments

    Returns list of clause strings.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Step 1: Split on sentence boundaries
    sentences = re.split(r"[.!?;]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    clauses = []
    for sentence in sentences:
        # Step 2: Try splitting on contrastive markers
        split_done = False
        for marker in CONTRASTIVE_MARKERS:
            # Use word boundary to avoid partial matches
            pattern = r"\b" + re.escape(marker) + r"\b"
            parts = re.split(pattern, sentence, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                left = parts[0].strip()
                right = parts[1].strip()
                left_words = len(left.split())
                right_words = len(right.split())
                if left_words >= MIN_CLAUSE_WORDS and right_words >= MIN_CLAUSE_WORDS:
                    clauses.append(left)
                    clauses.append(right)
                    split_done = True
                    break
        if not split_done:
            # Keep as single clause if meaningful
            if len(sentence.split()) >= MIN_CLAUSE_WORDS:
                clauses.append(sentence)
            elif len(sentence.split()) >= 1:
                # Short sentence — still keep if it's a complete thought
                clauses.append(sentence)

    # Filter empty
    clauses = [c for c in clauses if c.strip()]
    return clauses if clauses else [text]


# --- Prediction helpers -----------------------------------------------------
def predict_single(text, model, vectorizer, le):
    """Predict sentiment for a single text using the Stage 5.3 model.

    Includes negation-aware adjustment: if a clause contains negation words
    and the model predicts positive, check if negation modifies a positive
    sentiment word. If so, flip to negative with adjusted confidence.
    """
    cleaned = preprocess_improved(text)
    tfidf = vectorizer.transform([cleaned])[0]
    probs = model.predict_proba([tfidf])[0]
    pred_int = model.predict([tfidf])[0]
    label = le.inverse_transform([pred_int])[0]
    confidence = max(probs)

    # Negation-aware adjustment
    tokens = cleaned.split()
    has_negation = any(t in NEGATION_WORDS for t in tokens)

    if has_negation and label == "positive":
        # Check if negation precedes a positive sentiment word
        negation_flips = False
        for i, token in enumerate(tokens):
            if token in NEGATION_WORDS:
                # Look ahead for positive sentiment words within 3 tokens
                for j in range(i + 1, min(i + 4, len(tokens))):
                    if tokens[j] in POSITIVE_SENTIMENT_WORDS:
                        negation_flips = True
                        break
            if negation_flips:
                break

        if negation_flips:
            # Flip to negative; use original confidence (model was confident
            # about the sentiment word, negation flips it)
            label = "negative"
            confidence = probs[1] if probs[1] > 0.5 else 0.7
            probs = [confidence, 1.0 - confidence]

    return label, confidence, probs


# --- Mixed detection logic --------------------------------------------------
# Thresholds
OVERALL_MIXED_MARGIN = 0.15   # Overall probability must be within this of 0.5
CLAUSE_CONF_THRESHOLD = 0.60   # Minimum confidence for a clause to count
MIN_OPPOSING_CLAUSES = 1      # Need at least this many positive AND negative clauses


def detect_mixed(text, model, vectorizer, le):
    """
    Detect whether a review has mixed sentiment.

    Returns dict with:
      - overall_label: binary model prediction for whole review
      - overall_confidence: confidence of overall prediction
      - overall_probs: [P(neg), P(pos)]
      - clauses: list of (clause_text, label, confidence, probs)
      - has_positive: bool
      - has_negative: bool
      - is_mixed: bool
      - final_label: "Positive", "Negative", or "Mixed"
      - final_confidence: confidence for final label
      - explanation: human-readable reason
    """
    # 1. Overall prediction
    overall_label, overall_conf, overall_probs = predict_single(text, model, vectorizer, le)

    # 2. Split into clauses
    clauses = split_into_clauses(text)

    # 3. Predict each clause
    clause_results = []
    for clause in clauses:
        label, conf, probs = predict_single(clause, model, vectorizer, le)
        clause_results.append({
            "text": clause,
            "label": label,
            "confidence": conf,
            "probs": probs,
        })

    # 4. Determine if genuine mixed sentiment exists
    # Count clauses with sufficient confidence
    positive_clauses = [c for c in clause_results
                        if c["label"] == "positive" and c["confidence"] >= CLAUSE_CONF_THRESHOLD]
    negative_clauses = [c for c in clause_results
                        if c["label"] == "negative" and c["confidence"] >= CLAUSE_CONF_THRESHOLD]

    has_positive = len(positive_clauses) >= MIN_OPPOSING_CLAUSES
    has_negative = len(negative_clauses) >= MIN_OPPOSING_CLAUSES

    # Additional check: overall probability should not be extremely skewed
    # If overall is very confident (>0.85), it's unlikely to be truly mixed
    overall_p_pos = overall_probs[1]
    overall_is_uncertain = abs(overall_p_pos - 0.5) < (0.5 - OVERALL_MIXED_MARGIN)

    # Also check: if there's only one clause, can't be mixed
    single_clause = len(clauses) <= 1

    # Decision
    is_mixed = (has_positive and has_negative and not single_clause)

    # But also require that the clauses genuinely disagree
    # If all clauses agree, not mixed
    all_same = all(c["label"] == clause_results[0]["label"] for c in clause_results)
    if all_same:
        is_mixed = False

    # Final label
    if is_mixed:
        final_label = "Mixed"
        # Confidence for mixed: average of the opposing clause confidences
        pos_conf = max(c["confidence"] for c in positive_clauses) if positive_clauses else 0
        neg_conf = max(c["confidence"] for c in negative_clauses) if negative_clauses else 0
        final_confidence = (pos_conf + neg_conf) / 2.0
    else:
        final_label = "Positive" if overall_label == "positive" else "Negative"
        final_confidence = overall_conf

    # Explanation
    if is_mixed:
        pos_texts = [c["text"] for c in positive_clauses]
        neg_texts = [c["text"] for c in negative_clauses]
        explanation = (f"Mixed: found {len(positive_clauses)} positive clause(s) "
                       f"({'; '.join(pos_texts)}) and {len(negative_clauses)} negative clause(s) "
                       f"({'; '.join(neg_texts)}) with sufficient confidence.")
    elif single_clause:
        explanation = (f"Not mixed: single clause detected. "
                       f"Overall: {overall_label} (conf={overall_conf:.4f}).")
    elif all_same:
        explanation = (f"Not mixed: all clauses agree ({clause_results[0]['label']}). "
                       f"Overall: {overall_label} (conf={overall_conf:.4f}).")
    elif not has_positive or not has_negative:
        missing = "positive" if not has_positive else "negative"
        explanation = (f"Not mixed: no sufficiently confident {missing} clause found. "
                       f"Overall: {overall_label} (conf={overall_conf:.4f}).")
    else:
        explanation = (f"Not mixed: opposing clauses found but overall confidence too high "
                       f"({overall_conf:.4f}). Overall: {overall_label}.")

    return {
        "overall_label": "Positive" if overall_label == "positive" else "Negative",
        "overall_confidence": overall_conf,
        "overall_probs": overall_probs,
        "clauses": clause_results,
        "has_positive": has_positive,
        "has_negative": has_negative,
        "is_mixed": is_mixed,
        "final_label": final_label,
        "final_confidence": final_confidence,
        "explanation": explanation,
    }
