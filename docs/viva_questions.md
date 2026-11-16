# Viva Questions - Sentiment Analysis Project

**20+ Questions with Easy-to-Understand Answers**  
**Student Level:** Beginner B.Tech CSE (AI & ML)

---

## Basic Questions

### Q1. What is Sentiment Analysis?
**A:** Sentiment Analysis (also called Opinion Mining) is the process of determining whether a piece of text expresses a positive, negative, or neutral emotion. In this project, we classify customer reviews as either Positive or Negative.

### Q2. What is the problem statement of your project?
**A:** The problem is to build an automated system that can read customer reviews and classify their sentiment as Positive or Negative, without manual reading of each review.

### Q3. What is the objective of your project?
**A:** The objectives are to load review data, clean and preprocess text, perform EDA, extract features using TF-IDF, train a Logistic Regression model, evaluate its performance, and build a Streamlit web app for predictions.

### Q4. What dataset did you use?
**A:** We used a sample dataset (`dataset/reviews.csv`) containing customer reviews labeled as Positive or Negative. It has 60 reviews (30 positive, 30 negative). ⚠️ This is an educational sample dataset, not a real-world commercial dataset.

---

## Data Preprocessing Questions

### Q5. What is data preprocessing and why is it needed?
**A:** Data preprocessing is cleaning and preparing raw data before training a model. It is needed because raw text contains noise like punctuation, numbers, missing values, and duplicates that can confuse the ML model.

### Q6. How did you handle missing values?
**A:** We removed rows that had missing values in either the review text or sentiment column using `dropna()`.

### Q7. How did you handle duplicate reviews?
**A:** We removed duplicate reviews using `drop_duplicates()` so the same review doesn't appear multiple times in training data.

### Q8. What text cleaning steps did you perform?
**A:** We converted text to lowercase, removed punctuation marks, removed numbers, and normalized whitespace (removed extra spaces).

## Feature Extraction Questions

### Q10. What is TF-IDF Vectorizer?
**A:** TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical vectors. It gives higher weight to words that appear frequently in a document but rarely across all documents, helping identify important words.

### Q11. What does TF-IDF stand for?
**A:** TF = Term Frequency (how often a word appears in a document), IDF = Inverse Document Frequency (how rare a word is across all documents).

### Q12. What are stop words and why remove them?
**A:** Stop words are common words like "the", "is", "are", "a" that appear frequently but don't carry much meaning. Removing them reduces noise and focuses on important words.

### Q13. What is ngram_range=(1,2) in TF-IDF?
**A:** It means we consider both single words (unigrams) and pairs of consecutive words (bigrams) as features. This helps capture phrases like "not good" which have different sentiment than "good" alone.

## Algorithm Questions

### Q14. What algorithm did you use and why?
**A:** We used Logistic Regression because it is simple, interpretable, works well for text classification, trains quickly, and provides probability scores (confidence) for predictions.

### Q15. Is Logistic Regression used for regression or classification?
**A:** Despite its name, Logistic Regression is used for **classification**, not regression. It predicts the probability that an input belongs to a class.

### Q16. How does Logistic Regression make predictions?
**A:** It computes a weighted sum of features, applies the sigmoid function to get a probability (0 to 1), and classifies as Positive if probability > 0.5, otherwise Negative.

## Model Evaluation Questions

### Q18. What is train-test split and why is it important?
**A:** Train-test split divides data into training (80%) and testing (20%) sets. The model learns from training data and is evaluated on unseen test data to check how well it generalizes.

### Q19. What are Accuracy, Precision, Recall, and F1-Score?
**A:**
- **Accuracy**: Overall percentage of correct predictions
- **Precision**: Of all reviews predicted as Positive, how many were actually Positive
- **Recall**: Of all actual Positive reviews, how many were correctly identified
- **F1-Score**: Harmonic mean of Precision and Recall

### Q20. What is a Confusion Matrix?
**A:** A table showing True Positives, True Negatives, False Positives, and False Negatives. It helps visualize where the model makes mistakes.

### Q21. Why evaluate on test data and not training data?
**A:** Training data accuracy shows how well the model memorized the data. Test data accuracy shows how well the model performs on new, unseen data - which is the real measure of performance.

## Application Questions

### Q22. What is Streamlit and why did you use it?
**A:** Streamlit is a Python framework for building web applications quickly. We used it to create an easy-to-use interface where users can enter reviews and get instant sentiment predictions.

### Q23. What features does your web app have?
**A:** Text input box, Predict button, Positive/Negative result with emoji, confidence score, sample reviews, dataset statistics, model performance metrics, and error handling for empty inputs.

### Q24. How does your app make predictions?

## Technical Questions

### Q26. What are pickle files and why use them?
**A:** Pickle files (.pkl) are used to save Python objects (like trained models and vectorizers) to disk. This allows us to load them later without retraining the model.

### Q27. What is the difference between fit_transform and transform in TF-IDF?
**A:** `fit_transform` is used on training data to learn the vocabulary and create vectors. `transform` is used on test/new data to create vectors using the already-learned vocabulary.

### Q28. What is stratified train-test split?
**A:** Stratified split ensures that both Positive and Negative classes are represented in the same proportion in training and testing sets as in the original data.

### Q29. What is max_features=5000 in TF-IDF?
**A:** It limits the vocabulary to the top 5000 most important words, reducing the number of features and preventing the model from becoming too complex.

### Q30. What are the limitations of your project?
**A:** Small sample dataset (60 reviews), only binary classification (no Neutral), simple Logistic Regression algorithm, no handling of sarcasm or complex language nuances.

## General Questions

### Q31. Can this model be used for real products?
**A:** The current model is trained on a small sample dataset and is suitable for learning purposes. For real products, you would need a much larger, real-world dataset and more advanced models.

### Q32. What is overfitting and how do you prevent it?
**A:** Overfitting is when a model memorizes training data but fails on new data. We prevent it by using train-test split, evaluating on test data, and using a simple algorithm like Logistic Regression.

### Q33. How would you improve this project?
**A:** Use larger real-world datasets, add Neutral category, try deep learning models (LSTM, BERT), add multilingual support, and implement aspect-based sentiment analysis.

---

**Good luck with your viva! Remember to speak confidently and show the live demo.**
**A:** The app loads the trained model and TF-IDF vectorizer from pickle files, transforms the user's input text using the vectorizer, and passes it to the model for prediction.

### Q25. What happens if the user enters an empty review?
**A:** The app shows a warning message asking the user to enter a review text before predicting.

### Q17. What is the sigmoid function?
**A:** The sigmoid function converts any real number into a value between 0 and 1: σ(z) = 1 / (1 + e^(-z)). It is used to get probabilities.

### Q9. Why convert text to lowercase?
**A:** Lowercasing ensures that words like "Good" and "good" are treated as the same word, preventing the model from seeing them as different features.