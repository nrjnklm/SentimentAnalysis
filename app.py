"""
================================================================================
Sentiment Analysis of Movie Reviews - Streamlit Web Application
================================================================================

A beautiful, beginner-friendly Streamlit app for predicting sentiment
of movie reviews using a trained Logistic Regression model.

Supports: Positive, Negative, and Mixed sentiment detection.

Student Level: Beginner B.Tech CSE (AI & ML)
================================================================================
"""

import streamlit as st
import os
import sys
import re
import string
import json
import csv
import io
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_pipeline import TfidfVectorizer, LabelEncoder
from train_stage53 import LogisticRegressionSGD, preprocess_improved
from mixed_sentiment import load_model as load_mixed_model, detect_mixed

st.set_page_config(
    page_title="Sentiment Analysis - Movie Reviews",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        padding: 1.25rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0 0.5rem 0;
    }
    .positive-box {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .negative-box {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .mixed-box {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    .confidence-box {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.05rem;
        margin: 0.5rem 0 1rem 0;
    }
    .clause-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.95rem;
    }
    .section-spacer {
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
    }
    .stTextArea textarea {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_artifacts():
    """Load the Stage 5.3 100-epoch model, vectorizer, and label encoder."""
    model, vectorizer, label_encoder = load_mixed_model()
    return model, vectorizer, label_encoder


def predict_sentiment(review_text, model, vectorizer, label_encoder):
    """Predict sentiment for a given review text using Mixed detection."""
    result = detect_mixed(review_text, model, vectorizer, label_encoder)
    return result["final_label"], result["final_confidence"], result


def show_home_page(model, vectorizer, label_encoder):
    """Home page with prediction interface."""
    st.header("🔮 Predict Sentiment")

    if model is None or vectorizer is None or label_encoder is None:
        st.error("⚠️ Model files not found! Please train the model first by running: `python train_stage53.py`")
        st.info("The model files (sentiment_model_stage53.json, tfidf_vectorizer_stage53.json, label_encoder_stage53.json) should be in the `models/` folder.")
        return

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    review_text = st.text_area(
        "Enter a movie review:",
        key="review_input",
        placeholder="Type your review here... Example: 'This film was absolutely brilliant. I loved every minute of it.'",
        height=150,
        help="Enter any movie review text to analyze its sentiment."
    )

    def set_sample_text(text):
        st.session_state.review_input = text
        st.session_state.prediction_done = False

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    st.subheader("📝 Sample Reviews for Testing")

    col1, col2, col3 = st.columns(3)

    sample_reviews = [
        ("Positive", "This film was absolutely brilliant. I loved every minute of it."),
        ("Negative", "Terrible movie. Complete waste of time. Do not watch it."),
        ("Mixed", "The acting was excellent, but the story was boring."),
    ]

    for i, (label, text) in enumerate(sample_reviews):
        col = [col1, col2, col3][i]
        col.button(label, key=f"sample_{i}", on_click=set_sample_text, args=(text,), width="stretch")

    review_text = st.session_state.get("review_input", "") or ""

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

    if st.button("🎯 Predict Sentiment", type="primary", width="stretch"):
        if not review_text or review_text.strip() == "":
            st.warning("⚠️ Please enter a review text before predicting!")
            st.session_state.prediction_done = False
        else:
            st.session_state.prediction_done = True
            st.session_state.last_review = review_text.strip()
            sentiment, confidence, result = predict_sentiment(review_text, model, vectorizer, label_encoder)
            st.session_state.last_sentiment = sentiment
            st.session_state.last_confidence = confidence
            st.session_state.last_result = result

            # Record to prediction history (avoid duplicates on rerun)
            if "prediction_history" not in st.session_state:
                st.session_state.prediction_history = []
            # Only add if this is a new prediction (not a rerun of the same one)
            last_entry = st.session_state.prediction_history[-1] if st.session_state.prediction_history else None
            is_duplicate = (
                last_entry is not None
                and last_entry["review"] == review_text.strip()
                and last_entry["sentiment"] == sentiment
            )
            if not is_duplicate:
                st.session_state.prediction_history.append({
                    "review": review_text.strip(),
                    "sentiment": sentiment,
                    "confidence": round(confidence * 100, 2),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

    # Display prediction only if it was triggered for the current review
    if st.session_state.get("prediction_done", False) and st.session_state.get("last_review", "") == review_text.strip():
        sentiment = st.session_state.last_sentiment
        confidence = st.session_state.last_confidence
        result = st.session_state.last_result

        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)

        if sentiment == "Positive":
            st.markdown(f"""
            <div class="prediction-box positive-box">
                🟢 POSITIVE SENTIMENT
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        elif sentiment == "Negative":
            st.markdown(f"""
            <div class="prediction-box negative-box">
                🔴 NEGATIVE SENTIMENT
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box mixed-box">
                🟡 MIXED SENTIMENT
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="confidence-box">
            <strong>Confidence Score:</strong> {confidence:.2%}
        </div>
        """, unsafe_allow_html=True)

        if sentiment == "Mixed":
            st.subheader("Clause-Level Analysis")
            st.markdown("**Detected positive and negative clauses:**")
            for clause in result["clauses"]:
                label_icon = "🟢" if clause["label"] == "positive" else "🔴"
                st.markdown(f'<div class="clause-box">{label_icon} "{clause["text"]}" → {clause["label"].capitalize()} (conf={clause["confidence"]:.2%})</div>', unsafe_allow_html=True)
            st.info(result["explanation"])

        with st.expander("Overall Binary Model Prediction"):
            st.write(f"**Overall sentiment:** {result['overall_label']}")
            st.write(f"**Overall confidence:** {result['overall_confidence']:.2%}")
            st.write(f"**Overall probabilities:** Negative={result['overall_probs'][0]:.4f}, Positive={result['overall_probs'][1]:.4f}")

        st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
        st.info(f"**Review analyzed:** {review_text[:100]}{'...' if len(review_text) > 100 else ''}")


def load_dataset_stats():
    """Load dataset statistics for display from the IMDb train/test splits."""
    train_path = "dataset/imdb_train.csv"
    test_path = "dataset/imdb_test.csv"
    train_df = None
    test_df = None
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
    return train_df, test_df


def show_dataset_stats():
    """Dataset statistics page."""
    st.header("📊 Dataset Statistics")

    train_df, test_df = load_dataset_stats()

    if train_df is None or test_df is None:
        st.error("IMDb dataset files not found! Please ensure dataset/imdb_train.csv and dataset/imdb_test.csv exist.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Train Reviews", len(train_df))
    with col2:
        st.metric("Test Reviews", len(test_df))
    with col3:
        train_pos = (train_df["sentiment"] == "positive").sum()
        st.metric("Train Positive", int(train_pos))
    with col4:
        train_neg = (train_df["sentiment"] == "negative").sum()
        st.metric("Train Negative", int(train_neg))

    st.subheader("Sentiment Distribution — Training Set")
    sentiment_counts = train_df["sentiment"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor('#f0f2f6')
    ax.set_facecolor('#f0f2f6')
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette="Set2")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Sentiment Distribution — IMDb Training Set")
    for i, v in enumerate(sentiment_counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sample Reviews — Training Set")
    sample_html = train_df.head(10).to_html(index=False, classes="sample-table", border=0)
    st.markdown(sample_html, unsafe_allow_html=True)

    st.info("""
    **Dataset Information:**
    - IMDb Large Movie Review Dataset (official Stanford source).
    - Original training split: 25,000 reviews (12,500 positive / 12,500 negative).
    - Original test split: 25,000 reviews (12,500 positive / 12,500 negative), unchanged.
    - 123 overlapping reviews removed from training to prevent data leakage.
    - Effective training samples used: 24,877.
    - Real-world text with varied vocabulary — not synthetic template data.
    """)


def show_model_performance():
    """Model performance metrics page (values from Stage 5.3 model)."""
    st.header("📈 Model Performance")

    model, vectorizer, label_encoder = load_model_artifacts()

    if model is None:
        st.error("Model not trained yet! Please run `python train_stage53.py` first.")
        return

    st.subheader("Algorithm: Logistic Regression (SGD)")
    st.info("""
    **Logistic Regression with Stochastic Gradient Descent (SGD)** is a classification
    algorithm that predicts the probability of a binary outcome. In this project, it classifies
    movie reviews as Positive or Negative based on TF-IDF text features.

    **Mixed Sentiment Detection** is implemented as a post-processing layer:
    the review is split into clauses, each clause is evaluated by the binary model,
    and if both positive and negative clauses exist with sufficient confidence,
    the review is classified as Mixed.

    Negation words (e.g. "not") are preserved so the model correctly learns that
    "not good" is negative. Clause-level negation scope is also handled.
    """)

    st.subheader("Evaluation Metrics (on IMDb Test Data)")

    metrics_path = "models/metrics_stage53.json"
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    if metrics:
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}", help="Overall correctness on IMDb test set (binary)")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}", help="Of all positive predictions, how many were actually positive")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}", help="Of all actual positive reviews, how many were correctly identified")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.2%}", help="Harmonic mean of precision and recall")

        cm = metrics["confusion_matrix"]
        st.subheader("Confusion Matrix Details")
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            st.metric("True Negatives", f"{cm[0][0]:,}", help="Actual negative reviews correctly predicted as negative")
            st.metric("False Positives", f"{cm[0][1]:,}", help="Actual negative reviews incorrectly predicted as positive")
        with cm_col2:
            st.metric("False Negatives", f"{cm[1][0]:,}", help="Actual positive reviews incorrectly predicted as negative")
            st.metric("True Positives", f"{cm[1][1]:,}", help="Actual positive reviews correctly predicted as positive")

        st.info("""
        **Important:** These metrics were evaluated on the IMDb **binary** test set (Positive vs Negative).
        They are NOT three-class Mixed sentiment metrics. Mixed detection is a post-processing
        layer applied on top of the binary model. The confidence score shown on the Home page
        represents the model's raw prediction probability for the predicted class (or the
        average of opposing clause confidences for Mixed). It is NOT a guarantee of correctness.
        """)

        tc = metrics.get("training_config", {})
        if tc:
            st.subheader("Training Configuration")
            st.write(f"- **Epochs:** {tc.get('best_epochs', 'N/A')}")
            st.write(f"- **Learning rate:** {tc.get('lr', 'N/A')}")
            st.write(f"- **Max features:** {tc.get('vectorizer_max_features', 'N/A')}")
            st.write(f"- **N-gram range:** {tc.get('vectorizer_ngram_range', 'N/A')}")
            if tc.get('final_loss'):
                st.write(f"- **Final loss:** {tc.get('final_loss'):.4f}")
            st.write(f"- **Preprocessing:** {tc.get('preprocessing', 'N/A')}")
    else:
        st.info("metrics_stage53.json not found. Train the model first to populate performance metrics.")


def show_prediction_history():
    """Prediction History page — view and export session predictions."""
    st.header("📜 Prediction History")

    history = st.session_state.get("prediction_history", [])

    if not history:
        st.info("No predictions yet. Go to the Home page and analyze a review to build your history.")
        return

    # Build DataFrame for display
    df = pd.DataFrame(history)

    # Color-code sentiment
    def color_sentiment(val):
        if val == "Positive":
            return "background-color: #d4edda; color: #155724;"
        elif val == "Negative":
            return "background-color: #f8d7da; color: #721c24;"
        elif val == "Mixed":
            return "background-color: #fff3cd; color: #856404;"
        return ""

    st.subheader(f"Total predictions: {len(history)}")

    # Display table with colored sentiment
    styled_df = df.style.map(color_sentiment, subset=["sentiment"])
    st.dataframe(styled_df, width="stretch", hide_index=True)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total", len(history))
    with col2:
        pos_count = sum(1 for h in history if h["sentiment"] == "Positive")
        st.metric("Positive", pos_count)
    with col3:
        neg_count = sum(1 for h in history if h["sentiment"] == "Negative")
        st.metric("Negative", neg_count)
    with col4:
        mix_count = sum(1 for h in history if h["sentiment"] == "Mixed")
        st.metric("Mixed", mix_count)

    st.markdown("---")

    # CSV download
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["Review", "Sentiment", "Confidence", "Timestamp"])
    for entry in history:
        writer.writerow([entry["review"], entry["sentiment"], entry["confidence"], entry["timestamp"]])
    csv_data = csv_buffer.getvalue()

    col_dl, col_clear = st.columns(2)
    with col_dl:
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
            width="stretch",
        )

    with col_clear:
        if st.button("🗑️ Clear History", width="stretch"):
            st.session_state.confirm_clear = True

    # Confirmation dialog for clear
    if st.session_state.get("confirm_clear", False):
        st.warning("Are you sure you want to clear all prediction history?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, clear", type="primary", width="stretch"):
                st.session_state.prediction_history = []
                st.session_state.confirm_clear = False
                st.rerun()
        with col_no:
            if st.button("Cancel", width="stretch"):
                st.session_state.confirm_clear = False
                st.rerun()


def show_about_page():
    """About page with project information."""
    st.header("ℹ️ About This Project")

    st.markdown("""
    ### Project Title
    **Sentiment Analysis of Movie Reviews Using Machine Learning**

    ### Technologies Used
    - Python
    - Pandas (data manipulation)
    - Matplotlib & Seaborn (visualization)
    - Streamlit (web application)
    - TF-IDF Vectorizer (text feature extraction, pure Python)
    - Logistic Regression with SGD (classification algorithm, pure Python)

    ### Dataset
    **IMDb Large Movie Review Dataset** (official Stanford AI Lab source)
    - 50,000 real movie reviews: 25,000 positive / 25,000 negative
    - Official train/test split preserved (24,877 training / 25,000 test after dedup)
    - 123 overlapping reviews removed from training to prevent data leakage
    - Real-world text with varied vocabulary — not synthetic template data

    ### How It Works
    1. Movie reviews are loaded from the IMDb dataset CSV files
    2. Text is preprocessed (HTML tags stripped, lowercased, punctuation removed, whitespace collapsed)
    3. Negation words ("not", "never", "didn't") are preserved
    4. TF-IDF converts text into numerical features (unigrams + bigrams, 3000 features)
    5. Logistic Regression with SGD classifies sentiment (Positive/Negative)
    6. **Mixed detection**: Review is split into clauses at sentence boundaries and
       contrastive conjunctions (but, although, however). Each clause is evaluated
       independently. If both positive and negative clauses exist with sufficient
       confidence, the review is classified as Mixed.
    7. The Streamlit app provides an easy interface for predictions

    ### Student Level
    Beginner B.Tech CSE (AI & ML)

    ### Bug Fix Applied
    The original pipeline removed the word "not" as a stop word, so "Its not good at all"
    was reduced to just "good" and misclassified as Positive. Negation words are now
    preserved, labels are encoded explicitly (no reversal), and the dataset uses real
    IMDb reviews with correctly-labeled examples.

    ### Model Limitations
    - Trained with a pure-Python implementation (no scikit-learn) due to Windows
      Application Control blocking compiled ML library DLLs.
    - Confidence scores represent raw prediction probabilities, not guarantees of correctness.
    - The model may misclassify ambiguous, sarcastic, or neutral reviews.
    - Mixed detection uses heuristic clause splitting and negation scope (not full NLP parsing).
    - Neutral reviews (no clear sentiment) may be classified as Positive or Negative.
    - Evaluation accuracy on the IMDb test set: 87.97% (binary Positive/Negative).
    """)


def main():
    st.markdown('<h1 class="main-header">💬 Sentiment Analysis of Movie Reviews</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">A Machine Learning Project - Classify movie reviews as Positive, Negative, or Mixed</p>', unsafe_allow_html=True)

    model, vectorizer, label_encoder = load_model_artifacts()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction History", "Dataset Statistics", "Model Performance", "About"])

    if page == "Home":
        show_home_page(model, vectorizer, label_encoder)
    elif page == "Prediction History":
        show_prediction_history()
    elif page == "Dataset Statistics":
        show_dataset_stats()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "About":
        show_about_page()


if __name__ == "__main__":
    main()
