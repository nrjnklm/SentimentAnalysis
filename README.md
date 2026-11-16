# 💬 Sentiment Analysis of Customer Reviews Using Machine Learning

A complete beginner-friendly Machine Learning project that classifies customer reviews as **Positive** or **Negative** using Logistic Regression and TF-IDF vectorization. Includes a Streamlit web application for interactive predictions.

## 📁 Project Structure

```
SentimentAnalysisML/
├── app.py                      # Streamlit web application
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── dataset/
│   └── reviews.csv             # Customer review dataset
├── models/
│   ├── sentiment_model.pkl     # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
├── notebooks/
│   └── eda.py                  # Exploratory data analysis script
└── docs/
    ├── project_report.md       # Complete project report
    ├── presentation.md         # 14-slide presentation outline
    └── viva_questions.md       # 20+ viva questions with answers
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Install Dependencies

Open your terminal/command prompt in the `SentimentAnalysisML` folder and run:

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python train_model.py
```

This will:
- Load and clean the dataset
- Perform exploratory data analysis
- Train a Logistic Regression model
- Save the model and vectorizer to the `models/` folder
- Display evaluation metrics (accuracy, precision, recall, F1-score)

### Step 3: Run the Streamlit Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## 🎯 Features

- **Text Input**: Enter any customer review to analyze its sentiment
- **Sample Reviews**: Quick-test with predefined positive and negative reviews
- **Prediction with Confidence**: See the predicted sentiment and confidence score
- **Dataset Statistics**: View review counts and sentiment distribution
- **Model Performance**: View accuracy, precision, recall, F1-score, and confusion matrix
- **Error Handling**: Alerts for empty input and missing model files

## 📊 Dataset

The dataset (`dataset/reviews.csv`) contains customer reviews labeled as `positive` or `negative`.

**⚠️ Important Note**: This is a **sample/educational dataset** created for learning purposes. It is NOT a real-world commercial dataset. It contains 60 customer reviews (30 positive, 30 negative) for demonstration and learning.

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming language |
| Pandas | Data manipulation and loading |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning (Logistic Regression, TF-IDF, train-test split) |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Web application framework |

## 📈 Model Pipeline

1. **Data Loading**: Load reviews from CSV
2. **Data Cleaning**: Handle missing values, remove duplicates
3. **Text Preprocessing**: Lowercase, remove punctuation and numbers
4. **Train-Test Split**: 80% training, 20% testing
5. **TF-IDF Vectorization**: Convert text to numerical features
6. **Model Training**: Logistic Regression classifier
7. **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix

## 📖 Documentation

- `docs/project_report.md` - Complete project report for college submission
- `docs/presentation.md` - 14-slide presentation outline for viva
- `docs/viva_questions.md` - 20+ viva questions with easy-to-understand answers

## 👨‍💻 Author

B.Tech CSE (AI & ML) Student - Kalvium Machine Learning Final Project

## 📄 License

This project is created for educational purposes as part of the Kalvium ML curriculum.