"""
=================================================================================
Sentiment Analysis of Customer Reviews - Streamlit Web Application
=================================================================================

A beautiful, beginner-friendly Streamlit app for predicting sentiment
of customer reviews using a trained Logistic Regression model.

Student Level: Beginner B.Tech CSE (AI & ML)
=================================================================================
"""

import streamlit as st
import joblib
import os
import sys
import re
import string
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Sentiment Analysis - Customer Reviews",
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
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
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
    .confidence-box {
        background-color: #e2e3e5;
        color: #383d41;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    .info-box {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def clean_text(text):
    """Clean a single text string (same preprocessing as training)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_model_artifacts():
    """Load the trained model, TF-IDF vectorizer and label encoder.

    The SAME fitted artifacts produced by train_model.py are used here, so
    training-time and inference-time preprocessing/encoding are identical.
    """
    model_path = "models/sentiment_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"
    encoder_path = "models/label_encoder.pkl"

    if (not os.path.exists(model_path)
            or not os.path.exists(vectorizer_path)
            or not os.path.exists(encoder_path)):
        # Train model if files don't exist (e.g. fresh deployment)
        import subprocess
        subprocess.run([sys.executable, "train_model.py"], check=True)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    label_encoder = joblib.load(encoder_path)
    return model, vectorizer, label_encoder


def predict_sentiment(review_text, model, vectorizer, label_encoder):
    """Predict sentiment for a given review text.

    The input is cleaned with the same function used during training, then
    transformed with the fitted TF-IDF vectorizer and classified by the
    trained model. The encoded integer is mapped back through the saved
    LabelEncoder so Positive and Negative are never reversed.
    """
    cleaned = clean_text(review_text)
    review_tfidf = vectorizer.transform([cleaned])
    prediction_int = model.predict(review_tfidf)[0]
    probabilities = model.predict_proba(review_tfidf)[0]
    confidence = max(probabilities)
    prediction = label_encoder.inverse_transform([prediction_int])[0]
    return prediction, confidence


def show_home_page(model, vectorizer, label_encoder):
    """Home page with prediction interface."""
    st.header("🔮 Predict Sentiment")

    if model is None or vectorizer is None or label_encoder is None:
        st.error("⚠️ Model files not found! Please train the model first by running: `python train_model.py`")
        st.info("The model files (sentiment_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl) should be in the `models/` folder.")
        return

    review_text = st.text_area(
        "Enter a customer review:",
        placeholder="Type your review here... Example: 'The product is amazing! I love it so much.'",
        height=150,
        help="Enter any customer review text to analyze its sentiment."
    )

    st.subheader("📝 Sample Reviews for Testing")
    col1, col2, col3 = st.columns(3)

    sample_reviews = [
        ("Positive Sample 1", "The product is amazing! I love it so much. Best purchase I've made this year."),
        ("Positive Sample 2", "Excellent product! Easy to use and very durable. Five stars!"),
        ("Negative Sample 1", "Terrible quality, completely wasted my money. Would not recommend to anyone."),
        ("Negative Sample 2", "Worst product ever bought. It broke after two days of use. Very disappointed."),
        ("Negation: 'not good at all'", "Its not good at all"),
        ("Negation: 'does not work'", "This product does not work properly"),
    ]

    for i, (label, text) in enumerate(sample_reviews):
        col = [col1, col2, col3, col1, col2, col3][i]
        if col.button(label, key=f"sample_{i}"):
            st.session_state.review_input = text
            review_text = text

    if "review_input" in st.session_state:
        review_text = st.session_state.review_input

    if st.button("🎯 Predict Sentiment", type="primary", use_container_width=True):
        if not review_text or review_text.strip() == "":
            st.warning("⚠️ Please enter a review text before predicting!")
        else:
            sentiment, confidence = predict_sentiment(review_text, model, vectorizer, label_encoder)

            if sentiment.lower() == "positive":
                st.markdown(f"""
                <div class=\"prediction-box positive-box\">
                    ✅ POSITIVE SENTIMENT
                </div>
                """, unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
                <div class=\"prediction-box negative-box\">
                    ❌ NEGATIVE SENTIMENT
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class=\"confidence-box\">
                <strong>Confidence Score:</strong> {confidence:.2%}
            </div>
            """, unsafe_allow_html=True)

            st.info(f"**Review analyzed:** {review_text[:100]}{'...' if len(review_text) > 100 else ''}")



def load_dataset_stats():
    """Load dataset statistics for display."""
    dataset_path = "dataset/reviews.csv"
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        return df
    return None


def show_dataset_stats():
    """Dataset statistics page."""
    st.header("📊 Dataset Statistics")

    df = load_dataset_stats()

    if df is None:
        st.error("Dataset file not found!")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Reviews", len(df))
    with col2:
        positive_count = (df["sentiment"] == "positive").sum()
        st.metric("Positive Reviews", int(positive_count))
    with col3:
        negative_count = (df["sentiment"] == "negative").sum()
        st.metric("Negative Reviews", int(negative_count))

    st.subheader("Sentiment Distribution")
    sentiment_counts = df["sentiment"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax, palette="Set2")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Sentiment Distribution of Customer Reviews")
    for i, v in enumerate(sentiment_counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sample Reviews")
    sample_html = df.head(10).to_html(index=False, classes="sample-table", border=0)
    st.markdown(sample_html, unsafe_allow_html=True)

    st.info("""
    **Dataset Information:**
    - This is a balanced, correctly-labeled sentiment dataset.
    - Contains customer reviews labeled as Positive or Negative.
    - Negation examples (e.g. "not good at all") are included and labeled Negative.
    """)


def show_model_performance():
    """Model performance metrics page (values come from metrics.json produced by training)."""
    st.header("📈 Model Performance")

    model, vectorizer, label_encoder = load_model_artifacts()

    if model is None:
        st.error("Model not trained yet! Please run `python train_model.py` first.")
        return

    st.subheader("Algorithm: Logistic Regression")
    st.info("""
    **Logistic Regression** is a classification algorithm that predicts the probability
    of a binary outcome. In this project, it classifies customer reviews as Positive or Negative
    based on TF-IDF text features. Negation words (e.g. "not") are preserved so the model
    correctly learns that "not good" is negative.
    """)

    st.subheader("Evaluation Metrics (on Test Data)")

    cm_path = "confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix", use_column_width=True)
    else:
        st.info("Confusion matrix visualization not found. Train the model first.")

    metrics_path = "metrics.json"
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    if metrics:
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}", help="Overall correctness of the model")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}", help="Of all positive predictions, how many were actually positive")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}", help="Of all actual positive reviews, how many were correctly identified")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.2%}", help="Harmonic mean of precision and recall")
    else:
        st.info("metrics.json not found. Train the model first to populate performance metrics.")


def show_about_page():
    """About page with project information."""
    st.header("ℹ️ About This Project")

    st.markdown("""
    ### Project Title
    **Sentiment Analysis of Customer Reviews Using Machine Learning**

    ### Technologies Used
    - Python
    - Pandas & NumPy (data manipulation)
    - Scikit-learn (machine learning)
    - Matplotlib & Seaborn (visualization)
    - Streamlit (web application)
    - TF-IDF Vectorizer (text feature extraction)
    - Logistic Regression (classification algorithm)

    ### How It Works
    1. Customer reviews are loaded from a CSV dataset
    2. Text is cleaned (lowercase, remove punctuation/numbers)
    3. TF-IDF converts text into numerical features
    4. Logistic Regression classifies the sentiment
    5. The Streamlit app provides an easy interface for predictions

    ### Student Level
    Beginner B.Tech CSE (AI & ML)

    ### Bug Fix Applied
    The original pipeline removed the word "not" as a stop word, so "Its not good at all"
    was reduced to just "good" and misclassified as Positive. Negation words are now
    preserved, labels are encoded explicitly (no reversal), and the dataset was expanded
    to a large balanced set with correctly-labeled negation examples.
    """)


def main():
    st.markdown('<h1 class="main-header">💬 Sentiment Analysis of Customer Reviews</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">A Machine Learning Project - Classify Customer Reviews as Positive or Negative</p>', unsafe_allow_html=True)

    model, vectorizer, label_encoder = load_model_artifacts()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Dataset Statistics", "Model Performance", "About"])

    if page == "Home":
        show_home_page(model, vectorizer, label_encoder)
    elif page == "Dataset Statistics":
        show_dataset_stats()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "About":
        show_about_page()


if __name__ == "__main__":
    main()

