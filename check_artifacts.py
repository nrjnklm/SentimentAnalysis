import joblib
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Check label encoder
le_path = os.path.join(base_dir, "models", "label_encoder.pkl")
le = joblib.load(le_path)
print("=== Label Encoder ===")
print("Classes:", list(le.classes_))
for i, cls in enumerate(le.classes_):
    print(f"  {i} -> {cls}")

# Check model classes
model_path = os.path.join(base_dir, "models", "sentiment_model.pkl")
model = joblib.load(model_path)
print("\n=== Model ===")
print("Model classes_:", list(model.classes_))

# Check vectorizer
vec_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")
vectorizer = joblib.load(vec_path)
print("\n=== Vectorizer ===")
print("Type:", type(vectorizer).__name__)
print("Stop words sample:", list(vectorizer.get_stop_words())[:20] if hasattr(vectorizer, 'get_stop_words') else 'N/A')

# Test predictions
test_texts = [
    "very good",
    "excellent product",
    "I love this",
    "very bad",
    "terrible product",
    "not good at all",
    "Its not good at all",
]

from train_model import clean_text

print("\n=== Predictions ===")
for text in test_texts:
    cleaned = clean_text(text)
    tfidf = vectorizer.transform([cleaned])
    pred_int = model.predict(tfidf)[0]
    proba = model.predict_proba(tfidf)[0]
    pred_label = le.inverse_transform([pred_int])[0]
    confidence = max(proba)
    print(f"  {text!r:40} cleaned={cleaned!r:40} pred={pred_int} -> {pred_label:10} confidence={confidence:.4f} proba={proba}")
