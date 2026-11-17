import json
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer
with open(os.path.join(base_dir, "models", "sentiment_model.json"), "r") as f:
    model = json.load(f)
with open(os.path.join(base_dir, "models", "tfidf_vectorizer.json"), "r") as f:
    vectorizer = json.load(f)

vocab = vectorizer["vocabulary_"]
coef = model["coef_"]  # dict: {feature_idx: weight}
intercept = model["intercept_"]

# Find indices for "good", "very", "terrible", "bad", "excellent", "love"
for word in ["good", "very", "terrible", "bad", "excellent", "love", "not", "great"]:
    if word in vocab:
        idx = vocab[word]
        weight = coef.get(str(idx), coef.get(idx, 0))
        print(f"  '{word}' (idx={idx}): weight={weight:.6f}")
    else:
        print(f"  '{word}': NOT IN VOCABULARY")

# Check bigrams
for bigram in ["very good", "not good", "good at all", "terrible product", "excellent product"]:
    if bigram in vocab:
        idx = vocab[bigram]
        weight = coef.get(str(idx), coef.get(idx, 0))
        print(f"  '{bigram}' (idx={idx}): weight={weight:.6f}")
    else:
        print(f"  '{bigram}': NOT IN VOCABULARY")

print(f"\nIntercept: {intercept:.6f}")
print(f"Vocabulary size: {len(vocab)}")
print(f"Model coef_ keys type: {type(list(coef.keys())[0])}")
print(f"Sample coef_ keys: {list(coef.keys())[:5]}")