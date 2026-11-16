# Project Report: Sentiment Analysis of Customer Reviews Using Machine Learning

**Student Level:** Beginner B.Tech CSE (AI & ML)  
**Project Type:** Kalvium Machine Learning Final Project  
**Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, TF-IDF, Logistic Regression

---

## 1. Problem Statement

Customer reviews contain valuable feedback about products and services. However, manually reading and classifying hundreds or thousands of reviews is time-consuming and inefficient. Businesses need an automated system that can quickly analyze customer reviews and classify them as **Positive** or **Negative** to understand customer satisfaction and make informed decisions.

The problem is to build a Machine Learning system that can automatically classify the sentiment of customer reviews.

## 2. Abstract

This project presents a Sentiment Analysis system that classifies customer reviews into Positive or Negative categories using Machine Learning. The system uses TF-IDF (Term Frequency-Inverse Document Frequency) for feature extraction and Logistic Regression for classification. A Streamlit web application provides an easy-to-use interface for users to enter reviews and get instant sentiment predictions. The project includes complete data preprocessing, exploratory data analysis, model training, evaluation, and deployment.

## 3. Introduction

Sentiment analysis, also known as opinion mining, is the process of determining the emotional tone behind a piece of text. In e-commerce and customer service, understanding whether a review is positive or negative helps businesses:
- Identify product issues quickly
- Improve customer satisfaction
- Make data-driven decisions
- Prioritize customer support

This project implements a binary sentiment classifier using supervised Machine Learning.

## 4. Objectives

1. Load and explore a customer review dataset
2. Clean and preprocess text data (handle missing values, remove duplicates, clean text)
3. Perform exploratory data analysis with visualizations
4. Extract numerical features from text using TF-IDF Vectorizer
5. Train a Logistic Regression model for sentiment classification
6. Evaluate the model using accuracy, precision, recall, and F1-score
7. Build a Streamlit web application for real-time predictions
8. Create documentation for college presentation and viva

## 5. Existing System

Traditional sentiment analysis systems rely on manual classification, rule-based systems, and simple keyword matching. These systems are not scalable and cannot adapt to new language patterns.

## 6. Proposed System

The proposed system uses Machine Learning to automatically learn patterns from labeled data. It includes data collection, preprocessing, TF-IDF feature extraction, Logistic Regression training, prediction, and a Streamlit web interface.

## 7. Dataset Description


## 8. Data Preprocessing

### 8.1 Missing Value Handling
Rows with missing review text or sentiment labels are removed to ensure no null values in training data.

### 8.2 Duplicate Removal
Duplicate reviews are identified and removed to prevent bias from repeated examples.

### 8.3 Text Preprocessing
Each review is cleaned: lowercase conversion, punctuation removal, number removal, whitespace normalization.

### 8.4 Train-Test Split
80% of data used for training, 20% for testing, with stratified split to maintain class proportions.

## 9. Exploratory Data Analysis (EDA)

EDA includes sentiment distribution bar chart, review length statistics, word count analysis, and visualizations saved as PNG files.

Key findings: Balanced dataset with approximately equal positive and negative reviews.

## 10. Algorithm Explanation: Logistic Regression

Logistic Regression is a classification algorithm for binary problems. It predicts probability using sigmoid function. If probability > 0.5, classify as Positive; otherwise Negative.


## 13. Application Development

**Streamlit Web Application (`app.py`):**

The application provides:
- **Home Page**: Text input, Predict button, sentiment result with emoji, confidence score
- **Sample Reviews**: Predefined reviews for quick testing
- **Dataset Statistics**: Review counts, sentiment distribution chart
- **Model Performance**: Algorithm info, metrics, confusion matrix
- **About Page**: Project information and technologies used

**Features:** Beautiful UI with custom CSS, error handling for empty inputs, model file check, confidence score display, sample review buttons.

## 14. Conclusion

This project successfully demonstrates a complete Machine Learning pipeline for sentiment analysis including data loading, cleaning, text preprocessing, TF-IDF feature extraction, Logistic Regression training, model evaluation, and Streamlit web deployment.

## 15. Future Scope

1. Larger Dataset: Train on real-world datasets (Amazon reviews, IMDB reviews)
2. Multi-class Classification: Add Neutral sentiment category
3. Deep Learning: Use LSTM, BERT, or neural networks
4. Aspect-Based Sentiment: Sentiment for specific product features
5. Real-time Analysis: Connect to live review sources via APIs
6. Multilingual Support: Support reviews in multiple languages
7. Model Comparison: Compare with Naive Bayes, SVM, Random Forest

## 16. References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. Streamlit Documentation: https://docs.streamlit.io/
3. Pandas Documentation: https://pandas.pydata.org/
4. TF-IDF Vectorizer: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
5. Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

---

**End of Project Report**
**Why Logistic Regression?** Simple, interpretable, works well for text, fast training, provides confidence scores.

## 11. Model Training and Testing

TF-IDF Vectorization (max_features=5000, stop_words=english, ngram_range=(1,2)) converts text to features. Logistic Regression (max_iter=1000) is trained on 80% data and tested on 20%.

## 12. Evaluation Results

Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix. Actual values displayed after running `python train_model.py`.
**File:** `dataset/reviews.csv`  
**Format:** CSV with two columns: `review` and `sentiment`  
**Source:** This is a **sample/educational dataset** created for learning purposes. NOT a real-world commercial dataset. Contains 60 customer reviews (30 positive, 30 negative).

**⚠️ Important:** Do not claim this synthetic data is a real-world dataset.