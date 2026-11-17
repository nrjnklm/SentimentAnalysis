"""
=================================================================================
Sentiment Analysis of Customer Reviews - Model Training Script
=================================================================================

This script performs the complete machine learning pipeline:
1. Loads the customer reviews dataset from CSV
2. Cleans and preprocesses the text data
3. Handles missing values and removes duplicates
4. Performs exploratory data analysis (EDA)
5. Encodes sentiment labels (negative=0, positive=1) via LabelEncoder
6. Extracts features using TF-IDF Vectorizer (negation words are PRESERVED)
7. Trains a Logistic Regression model
8. Evaluates the model using accuracy, precision, recall, F1-score
9. Saves the trained model, vectorizer, label encoder and metrics JSON

Student Level: Beginner B.Tech CSE (AI & ML)
Project: Sentiment Analysis of Customer Reviews Using Machine Learning
=================================================================================
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_pipeline import train_and_save

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "reviews.csv")

def main():
    """Load the dataset from a CSV file."""
    print(f"[1/9] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Label distribution:\n{df['sentiment'].value_counts().to_dict()}")
    return df


def clean_text(text):
    """Clean a single text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_data(df):
    """Clean the entire dataset."""
    print("\n[2/9] Cleaning data...")
    missing_before = df.isnull().sum().sum()
    print(f"   Missing values before cleaning: {missing_before}")
    df = df.dropna(subset=["review", "sentiment"])
    duplicates_before = len(df)
    df = df.drop_duplicates(subset=["review"])
    duplicates_removed = duplicates_before - len(df)
    print(f"   Duplicate rows removed: {duplicates_removed}")
    df["review_clean"] = df["review"].apply(clean_text)
    df = df[df["review_clean"].str.len() > 0]
    print(f"   Dataset shape after cleaning: {df.shape}")
    return df


def perform_eda(df):
    """Perform exploratory data analysis and save visualizations."""
    print("\n[3/9] Performing Exploratory Data Analysis...")
    sentiment_counts = df["sentiment"].value_counts()
    print(f"   Sentiment distribution:\n{sentiment_counts}")
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x="sentiment", palette="Set2", order=[NEGATIVE_LABEL, POSITIVE_LABEL])
    plt.title("Sentiment Distribution of Customer Reviews", fontsize=14)
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Number of Reviews", fontsize=12)
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png", dpi=300)
    plt.close()
    print("   Sentiment distribution plot saved as 'sentiment_distribution.png'")

    df["review_length"] = df["review_clean"].apply(len)
    print(f"   Average review length: {df['review_length'].mean():.2f} characters")
    print(f"   Shortest review: {df['review_length'].min()} characters")
    print(f"   Longest review: {df['review_length'].max()} characters")


def encode_labels(df):
    """Encode sentiment labels using LabelEncoder (negative=0, positive=1)."""
    print("\n[4/9] Encoding sentiment labels...")
    le = LabelEncoder()
    df["sentiment_encoded"] = le.fit_transform(df["sentiment"])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"   Label mapping: {mapping}")
    print(f"   Classes: {list(le.classes_)}")
    return df, le


def split_data(df):
    """Split the dataset into training and testing sets (stratified)."""
    print("\n[5/9] Splitting data into train and test sets...")
    X = df["review_clean"]
    y = df["sentiment_encoded"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")
    return X_train, X_test, y_train, y_test


def extract_features(X_train, X_test):
    """Convert text data into TF-IDF feature vectors (negation words preserved)."""
    print("\n[6/9] Extracting features using TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words=CUSTOM_STOP_WORDS,
        ngram_range=(1, 2),
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"   Training features shape: {X_train_tfidf.shape}")
    print(f"   Testing features shape:  {X_test_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, vectorizer



def train_model(X_train_tfidf, y_train):
    """Train a Logistic Regression model."""
    print("\n[7/9] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)
    print("   Model training completed!")
    return model


def evaluate_model(model, X_test_tfidf, y_test, le):
    """Evaluate the trained model on test data and save metrics/confusion matrix."""
    print("\n[8/9] Evaluating model on test data...")
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"\n   Confusion Matrix:\n{cm}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title("Confusion Matrix - Sentiment Analysis", fontsize=14)
    plt.xlabel("Predicted Sentiment", fontsize=12)
    plt.ylabel("Actual Sentiment", fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()
    print("   Confusion matrix plot saved as 'confusion_matrix.png'")
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"   Metrics saved to '{METRICS_PATH}'")
    return metrics, y_pred


def save_artifacts(model, vectorizer, le):
    """Save the trained model, TF-IDF vectorizer, and label encoder as pickles."""
    print("\n[9/9] Saving model and vectorizer...")
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"   Model saved to: {MODEL_PATH}")
    print(f"   Vectorizer saved to: {VECTORIZER_PATH}")
    print(f"   Label encoder saved to: {LABEL_ENCODER_PATH}")


def predict_clean(text, model, vectorizer, le):
    """Clean text then predict sentiment label (string)."""
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    pred_int = model.predict(tfidf)[0]
    return le.inverse_transform([pred_int])[0]


def main():
    """Main function to run the complete training pipeline."""
    print("=" * 70)
    print("SENTIMENT ANALYSIS - MODEL TRAINING PIPELINE")
    print("=" * 70)
    df = load_data(DATASET_PATH)
    df = clean_data(df)
    perform_eda(df)
    df, le = encode_labels(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    model = train_model(X_train_tfidf, y_train)
    metrics, y_pred = evaluate_model(model, X_test_tfidf, y_test, le)
    save_artifacts(model, vectorizer, le)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nFinal Evaluation Results (on test data):")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-Score : {metrics['f1_score']:.4f}")

    print("\nSanity-check predictions (model must NOT hardcode these):")
    all_ok = True
    for text, expected in SAMPLE_REVIEWS:
        pred = predict_clean(text, model, vectorizer, le)
        ok = (pred == expected)
        all_ok = all_ok and ok
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {text!r:45} expected={expected:9} predicted={pred}")
    if all_ok:
        print("\nAll sanity-check reviews predicted correctly.")
    else:
        print("\nWARNING: some sanity-check reviews were NOT predicted correctly!")
    print("\nYou can now run the Streamlit app with: streamlit run app.py")


if __name__ == "__main__":
    main()

