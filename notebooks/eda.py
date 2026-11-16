"""
===============================================================================
Exploratory Data Analysis (EDA) - Sentiment Analysis Project
===============================================================================

This script performs detailed exploratory data analysis on the customer reviews
dataset. It generates visualizations and summary statistics to help understand
the data before model training.

Student Level: Beginner B.Tech CSE (AI & ML)
===============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re

DATASET_PATH = "../dataset/reviews.csv"


def load_data():
    """Load the dataset."""
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    return df


def clean_text(text):
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def basic_statistics(df):
    """Print basic statistics about the dataset."""
    print("\n" + "=" * 60)
    print("BASIC DATASET STATISTICS")
    print("=" * 60)

    print(f"\nTotal number of reviews: {len(df)}")
    print(f"\nSentiment distribution:")
    print(df["sentiment"].value_counts())

    print(f"\nSentiment percentages:")
    print(df["sentiment"].value_counts(normalize=True).mul(100).round(2))

    # Check for missing values
    print(f"\nMissing values per column:")
    print(df.isnull().sum())

    # Check for duplicates
    print(f"\nDuplicate rows: {df.duplicated().sum()}")


def text_statistics(df):
    """Analyze text length statistics."""
    print("\n" + "=" * 60)
    print("TEXT LENGTH STATISTICS")
    print("=" * 60)

    df["review_clean"] = df["review"].apply(clean_text)
    df["review_length"] = df["review_clean"].apply(len)
    df["word_count"] = df["review_clean"].apply(lambda x: len(x.split()))

    print(f"\nReview length (characters):")
    print(f"  Average: {df['review_length'].mean():.2f}")
    print(f"  Minimum: {df['review_length'].min()}")
    print(f"  Maximum: {df['review_length'].max()}")
    print(f"  Median:  {df['review_length'].median()}")

    print(f"\nWord count per review:")
    print(f"  Average: {df['word_count'].mean():.2f}")
    print(f"  Minimum: {df['word_count'].min()}")
    print(f"  Maximum: {df['word_count'].max()}")
    print(f"  Median:  {df['word_count'].median()}")

    # Text length by sentiment
    print(f"\nAverage review length by sentiment:")
    print(df.groupby("sentiment")["review_length"].mean().round(2))

    return df


def plot_visualizations(df):
    """Create and save visualizations."""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    plt.style.use("default")

    # 1. Sentiment distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="sentiment", palette="Set2")
    plt.title("Distribution of Sentiment Labels", fontsize=16, fontweight="bold")
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig("sentiment_distribution.png", dpi=300)
    plt.close()
    print("  Saved: sentiment_distribution.png")

    # 2. Review length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="review_length", hue="sentiment",
                 bins=20, palette="Set1", alpha=0.7)
    plt.title("Review Length Distribution by Sentiment", fontsize=16, fontweight="bold")
    plt.xlabel("Review Length (characters)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.savefig("review_length_distribution.png", dpi=300)
    plt.close()
    print("  Saved: review_length_distribution.png")

    # 3. Word count by sentiment (box plot)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="sentiment", y="word_count", palette="Set3")
    plt.title("Word Count Distribution by Sentiment", fontsize=16, fontweight="bold")
    plt.xlabel("Sentiment", fontsize=12)
    plt.ylabel("Word Count", fontsize=12)
    plt.tight_layout()
    plt.savefig("word_count_boxplot.png", dpi=300)
    plt.close()
    print("  Saved: word_count_boxplot.png")

    print("\nAll visualizations saved successfully!")


def main():
    """Main EDA function."""
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS - SENTIMENT ANALYSIS")
    print("=" * 60)

    df = load_data()
    basic_statistics(df)
    df = text_statistics(df)
    plot_visualizations(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()