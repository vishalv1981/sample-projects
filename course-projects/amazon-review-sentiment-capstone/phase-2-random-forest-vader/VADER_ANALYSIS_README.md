# Amazon Fine Food Reviews Sentiment Analysis

## Overview

This project analyzes the **Amazon Fine Food Reviews** dataset, which contains over 568,000 reviews of fine foods from Amazon. The objective of this analysis is to determine whether a review is positive or negative using sentiment analysis and compare it with the actual rating provided by users.

The dataset spans from **October 1999 to October 2012** and includes reviews with attributes like the product ID, user ID, review text, and helpfulness ratings.

The project uses **VADER Sentiment Analysis** to evaluate the sentiment of review text and compare the results with the actual review scores provided in the dataset.

## Dataset

- **Source:** [Kaggle: Amazon Fine Food Reviews](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- **Number of Reviews:** 568,454
- **Timespan:** October 1999 - October 2012
- **Attributes:**
  - `Id`: Unique identifier for each review.
  - `ProductId`: Unique identifier for the product.
  - `UserId`: Unique identifier for the user.
  - `ProfileName`: Name of the user.
  - `HelpfulnessNumerator`: Number of users who found the review helpful.
  - `HelpfulnessDenominator`: Total number of users who rated the helpfulness of the review.
  - `Score`: The rating given by the user (1-5).
  - `Time`: Timestamp of the review.
  - `Summary`: A brief summary of the review.
  - `Text`: The full review text.

## Objective

The primary objective is to classify whether a review is positive or negative based on the **review text** and compare this with the actual rating provided by the users (1-5 scale). Reviews with a score of 3 are treated as neutral and ignored.

- **Positive Review:** Score of 4 or 5.
- **Negative Review:** Score of 1 or 2.

## Methodology

### 1. Data Loading
The dataset is loaded from an SQLite database, which provides an efficient way to query and manipulate the data.

### 2. Data Preprocessing
- Reviews with a score of 3 are ignored.
- Reviews with scores of 1 or 2 are labeled as **negative**.
- Reviews with scores of 4 or 5 are labeled as **positive**.

### 3. Sentiment Analysis
The **VADER Sentiment Analysis tool** is used to analyze the sentiment of each review's text. VADER provides a compound score that indicates the overall sentiment polarity of the text. If the compound score is greater than 0.05, the review is classified as **positive**. If the score is below -0.05, it is classified as **negative**.

### 4. Evaluation
The VADER sentiment results are compared to the actual ratings from the dataset. Various metrics such as precision, recall, and F1-score are calculated to measure the performance of the sentiment analysis.

## Results

- The notebook includes visualizations, including a **confusion matrix** and **classification report**, to show how well the sentiment analysis performed in comparison to actual user ratings.
- **Precision**, **recall**, and **F1-score** are used to evaluate the performance of the sentiment classification.

## Running the Notebook

### Prerequisites
The following libraries are required to run the notebook:
- `numpy`
- `pandas`
- `matplotlib`
- `sqlite3`
- `nltk`
- `vaderSentiment`

You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib nltk vaderSentiment
