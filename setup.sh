# Streamlit Community Cloud setup script
# This file trains the ML model on first deployment

# Install dependencies
pip install -r requirements.txt

# Train the model (creates sentiment_model.pkl and tfidf_vectorizer.pkl)
python train_model.py