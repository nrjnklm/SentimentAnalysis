"""
Pure-Python ML pipeline for sentiment analysis.
Used when NumPy/SciPy/Scikit-Learn are unavailable (e.g., blocked DLLs).

Provides:
  - TfidfVectorizer: fit/transform text to TF-IDF feature vectors
  - LogisticRegression: binary classification via gradient descent
  - LabelEncoder: encode/decode sentiment labels

All artifacts are saved as JSON so they can be loaded without NumPy.
"""

import os
import json
import re
import string
import math
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Label encoding (EXPLICIT and consistent) ---------------------------------
# negative -> 0, positive -> 1
LABEL_TO_INT = {"negative": 0, "positive": 1}
INT_TO_LABEL = {0: "negative", 1: "positive"}

# --- Negation words preserved (NOT removed as stop words) ---------------------
NEGATION_WORDS = {
    "not", "no", "nor",
    "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "cant", "couldnt", "shouldnt", "wont", "wouldnt", "hasnt", "havent", "hadnt",
    "mustnt", "neednt", "mightn",
}

# sklearn ENGLISH_STOP_WORDS minus negation words
CUSTOM_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some",
    "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours",
    "yourself", "yourselves",
} - NEGATION_WORDS - {"very"}


def clean_text(text):
    """Clean a single text string (same preprocessing as training)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text):
    """Split cleaned text into tokens, removing stop words but keeping negations."""
    tokens = text.split()
    return [t for t in tokens if t not in CUSTOM_STOP_WORDS]


# --- TF-IDF Vectorizer --------------------------------------------------------
class TfidfVectorizer:
    """Minimal TF-IDF vectorizer (unigram + bigram, sparse output)."""

    def __init__(self, max_features=1000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_ = {}          # term -> index
        self.idf_ = {}                 # term -> idf value
        self.n_features_in_ = 0
        self._doc_count = 0

    def _get_ngrams(self, tokens):
        ngrams = []
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i:i + n]))
        return ngrams

    def fit(self, texts):
        """Learn vocabulary and IDF from a list of cleaned texts."""
        doc_freq = {}
        self._doc_count = len(texts)

        for text in texts:
            tokens = _tokenize(text)
            ngrams = self._get_ngrams(tokens)
            seen = set()
            for term in ngrams:
                if term not in seen:
                    doc_freq[term] = doc_freq.get(term, 0) + 1
                    seen.add(term)

        sorted_terms = sorted(doc_freq.items(), key=lambda x: (-x[1], x[0]))
        if self.max_features and len(sorted_terms) > self.max_features:
            sorted_terms = sorted_terms[:self.max_features]

        self.vocabulary_ = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
        self.idf_ = {}
        for term, df in sorted_terms:
            self.idf_[term] = math.log((self._doc_count + 1) / (df + 1)) + 1.0

        self.n_features_in_ = len(self.vocabulary_)
        return self

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def transform(self, texts):
        """Transform texts into sparse TF-IDF vectors (list of dicts: {idx: value})."""
        vectors = []
        for text in texts:
            tokens = _tokenize(text)
            ngrams = self._get_ngrams(tokens)
            tf = {}
            total = len(ngrams)
            if total == 0:
                vectors.append({})
                continue
            for term in ngrams:
                tf[term] = tf.get(term, 0) + 1
            vec = {}
            for term, count in tf.items():
                if term in self.vocabulary_:
                    idx = self.vocabulary_[term]
                    tf_val = count / total
                    vec[idx] = tf_val * self.idf_[term]
            vectors.append(vec)
        return vectors

    def to_dict(self):
        return {
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "vocabulary_": self.vocabulary_,
            "idf_": self.idf_,
            "n_features_in_": self.n_features_in_,
            "_doc_count": self._doc_count,
        }

    @classmethod
    def from_dict(cls, d):
        v = cls(max_features=d["max_features"], ngram_range=tuple(d["ngram_range"]))
        v.vocabulary_ = d["vocabulary_"]
        v.idf_ = d["idf_"]
        v.n_features_in_ = d["n_features_in_"]
        v._doc_count = d["_doc_count"]
        return v


# --- Logistic Regression ------------------------------------------------------
class LogisticRegression:
    """Binary logistic regression trained with gradient descent (L2 regularized).
    Works with sparse feature vectors (dicts: {feature_idx: value})."""

    def __init__(self, lr=0.5, n_iter=800, C=1.0, random_state=42):
        self.lr = lr
        self.n_iter = n_iter
        self.C = C
        self.random_state = random_state
        self.coef_ = None       # dict: {feature_idx: weight}
        self.intercept_ = None
        self.classes_ = None
        self.n_features_in_ = None

    @staticmethod
    def _sigmoid(z):
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

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

        for it in range(self.n_iter):
            grad_w = {j: 0.0 for j in range(self.n_features_in_)}
            grad_b = 0.0
            for i in range(n):
                row = X[i]
                z = self.intercept_ + sum(self.coef_[j] * v for j, v in row.items())
                p = self._sigmoid(z)
                error = p - y[i]
                grad_b += error
                for j, v in row.items():
                    grad_w[j] += error * v
            for j in range(self.n_features_in_):
                grad_w[j] = grad_w[j] / n + alpha * self.coef_[j]
            grad_b = grad_b / n
            self.intercept_ -= self.lr * grad_b
            for j in range(self.n_features_in_):
                self.coef_[j] -= self.lr * grad_w[j]
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
            "coef_": self.coef_,
            "intercept_": self.intercept_,
            "classes_": self.classes_,
            "n_features_in_": self.n_features_in_,
        }

    @classmethod
    def from_dict(cls, d):
        m = cls(lr=d["lr"], n_iter=d["n_iter"], C=d["C"], random_state=d["random_state"])
        m.coef_ = {int(k): v for k, v in d["coef_"].items()}
        m.intercept_ = d["intercept_"]
        m.classes_ = d["classes_"]
        m.n_features_in_ = d["n_features_in_"]
        return m


# --- Label Encoder ------------------------------------------------------------
class LabelEncoder:
    def __init__(self):
        self.classes_ = ["negative", "positive"]

    def fit(self, y):
        classes = sorted(set(y))
        self.classes_ = classes
        return self

    def transform(self, y):
        mapping = {c: i for i, c in enumerate(self.classes_)}
        return [mapping[c] for c in y]

    def inverse_transform(self, y):
        return [self.classes_[i] for i in y]

    def to_dict(self):
        return {"classes_": self.classes_}

    @classmethod
    def from_dict(cls, d):
        le = cls()
        le.classes_ = d["classes_"]
        return le


# --- Artifact save/load -------------------------------------------------------
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.json")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.json")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.json")


def save_artifacts(model, vectorizer, label_encoder):
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
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
    """Clean text then predict sentiment label (string)."""
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])[0]
    probs = model.predict_proba([tfidf])[0]
    pred_int = model.predict([tfidf])[0]
    label = le.inverse_transform([pred_int])[0]
    confidence = max(probs)
    return label, confidence, probs
    return label, confidence, probs


# --- Training pipeline --------------------------------------------------------
import csv


def load_data(file_path):
    """Load dataset from CSV using only the standard library."""
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    class DF:
        def __init__(self, rows):
            self._rows = rows
            self.columns = list(rows[0].keys()) if rows else []

        def __getattr__(self, name):
            if name in ("_rows", "columns"):
                return super().__getattribute__(name)
            return [r[name] for r in self._rows]

        def __getitem__(self, name):
            return [r[name] for r in self._rows]

        def __setitem__(self, name, values):
            for r, v in zip(self._rows, values):
                r[name] = v

        def dropna(self, subset=None):
            if subset is None:
                return self
            filtered = [r for r in self._rows if all(r.get(k) not in (None, "") for k in subset)]
            return DF(filtered)

        def drop_duplicates(self, subset=None):
            if subset is None:
                return self
            seen = set()
            filtered = []
            for r in self._rows:
                key = tuple(r.get(k) for k in subset)
                if key not in seen:
                    seen.add(key)
                    filtered.append(r)
            return DF(filtered)

        def __len__(self):
            return len(self._rows)

        @property
        def shape(self):
            return (len(self._rows), len(self.columns))

    df = DF(rows)
    print(f"   Dataset shape: {df.shape}")
    dist = {}
    for r in df._rows:
        s = r.get("sentiment", "")
        dist[s] = dist.get(s, 0) + 1
    print(f"   Label distribution: {dist}")
    return df


def clean_data(df):
    df = df.dropna(subset=["review", "sentiment"])
    df = df.drop_duplicates(subset=["review"])
    df["review_clean"] = [clean_text(r["review"]) for r in df._rows]
    df._rows = [r for r in df._rows if r.get("review_clean", "").strip()]
    print(f"   Dataset shape after cleaning: {df.shape}")
    return df


def encode_labels(df):
    le = LabelEncoder()
    sentiments = df["sentiment"]
    le.fit(sentiments)
    df["sentiment_encoded"] = le.transform(sentiments)
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"   Label mapping: {mapping}")
    print(f"   Classes: {list(le.classes_)}")
    return df, le


def split_data(df, test_size=0.2, random_state=42):
    X = df["review_clean"]
    y = df["sentiment_encoded"]
    pos_idx = [i for i, v in enumerate(y) if v == 1]
    neg_idx = [i for i, v in enumerate(y) if v == 0]
    import random
    rng = random.Random(random_state)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_pos_test = max(1, int(len(pos_idx) * test_size))
    n_neg_test = max(1, int(len(neg_idx) * test_size))
    test_idx = set(pos_idx[:n_pos_test] + neg_idx[:n_neg_test])
    train_idx = [i for i in range(len(y)) if i not in test_idx]
    test_idx = sorted(test_idx)
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples:  {len(X_test)}")
    return X_train, X_test, y_train, y_test


def train_and_save(dataset_path):
    print("=" * 70)
    print("SENTIMENT ANALYSIS - MODEL TRAINING PIPELINE (Pure Python)")
    print("=" * 70)

    df = load_data(dataset_path)

    df = clean_data(df)

    df, le = encode_labels(df)

    X_train, X_test, y_train, y_test = split_data(df)

    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"   Training features shape: {len(X_train_tfidf)} x {vectorizer.n_features_in_}")
    print(f"   Testing features shape:  {len(X_test_tfidf)} x {vectorizer.n_features_in_}")

    model = LogisticRegression(lr=0.5, n_iter=400, C=5.0, random_state=42)
    model.fit(X_train_tfidf, y_train)
    print("   Model training completed!")

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    correct = sum(1 for a, b in zip(y_test, y_pred) if a == b)
    accuracy = correct / len(y_test)
    tp = sum(1 for a, b in zip(y_test, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_test, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_test, y_pred) if a == 1 and b == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall   : {recall:.4f}")
    print(f"   F1-Score : {f1:.4f}")

    save_artifacts(model, vectorizer, le)

    # Sanity checks
    SAMPLE_REVIEWS = [
        ("Its not good at all", "negative"),
        ("This product is terrible", "negative"),
        ("I hate this product", "negative"),
        ("This product is excellent", "positive"),
        ("I love this product", "positive"),
        ("Very happy with my purchase", "positive"),
    ]
    print("\nSanity-check predictions (model must NOT hardcode these):")
    all_ok = True
    for text, expected in SAMPLE_REVIEWS:
        pred, conf, _ = predict_clean(text, model, vectorizer, le)
        ok = (pred == expected)
        all_ok = all_ok and ok
        flag = "OK" if ok else "FAIL"
        print(f"  [{flag}] {text!r:45} expected={expected:9} predicted={pred} confidence={conf:.4f}")
    if all_ok:
        print("\nAll sanity-check reviews predicted correctly.")
    else:
        print("\nWARNING: some sanity-check reviews were NOT predicted correctly!")

    print("\nTRAINING COMPLETED SUCCESSFULLY!")
    print("You can now run the Streamlit app with: streamlit run app.py")


if __name__ == "__main__":
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "reviews.csv")
    train_and_save(DATASET_PATH)
