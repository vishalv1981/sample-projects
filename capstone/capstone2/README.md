# Logistic Regression for Amazon Fine Food Reviews

## Overview

This project performs sentiment analysis using logistic regression on the Amazon Fine Food Reviews dataset. The main objective is to classify user reviews as either positive or negative based on the review text. The dataset contains over 500,000 reviews, and the sentiment is derived from the `Score` column, which is converted into binary classes (positive or negative).

## Dataset

- **Source**: [Amazon Fine Food Reviews Dataset on Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews)
- **Number of Reviews**: 568,454
- **Number of Users**: 256,059
- **Number of Products**: 74,258
- **Review Time Period**: October 1999 - October 2012

### Data Fields:
1. **Id**: Row ID
2. **ProductId**: Unique identifier for each product
3. **UserId**: Unique identifier for each user
4. **ProfileName**: Name of the user
5. **HelpfulnessNumerator**: Number of users who found the review helpful
6. **HelpfulnessDenominator**: Total number of users who rated the review's helpfulness
7. **Score**: Rating between 1 and 5
8. **Time**: Timestamp of the review
9. **Summary**: Brief summary of the review
10. **Text**: Full text of the review

### Objective

The goal is to determine whether a review is positive or negative based on its score:
- **Positive**: Reviews with a score of 4 or 5.
- **Negative**: Reviews with a score of 1 or 2.
- **Neutral**: Reviews with a score of 3 are ignored.

## Key Steps

### 1. Data Preprocessing
- **Cleaning**: Removed null values and irrelevant fields.
- **Binary Labeling**: Converted the `Score` into positive/negative sentiment.
- **Text Processing**:
  - Applied tokenization, stopword removal, and stemming/lemmatization.
  - Used Bag of Words (BoW) and TF-IDF to convert text into numerical format.

### 2. Classifiers and Models

#### 2.1 Logistic Regression
- **Model Overview**: Logistic regression is a linear classifier used in this project to predict the sentiment of reviews (positive or negative). It was chosen due to its simplicity and effectiveness for binary classification tasks.
- **Implementation**:
  - The review text was vectorized using Bag of Words and TF-IDF methods.
  - Logistic Regression was trained using these vectorized features, with cross-validation applied for robust performance.
- **Hyperparameter Tuning**:
  - The model’s regularization parameter `C` was optimized using grid search.
  - Various values of `C` were tested to find the best trade-off between bias and variance.

#### 2.2 Random Forest (Optional Model for Comparison)
- **Model Overview**: Random Forest is an ensemble classifier that operates by building multiple decision trees during training and outputting the class that is the majority vote of the trees. This was used as an additional classifier to compare its performance with logistic regression.
- **Implementation**:
  - Random Forest was trained using the same vectorized features.
  - Cross-validation was applied to check for overfitting and to ensure model generalization.
- **Hyperparameter Tuning**:
  - Parameters like the number of estimators (trees) and the maximum depth of the trees were tuned using grid search.

### 3. Model Building
- **Logistic Regression**: 
  - Chosen for its simplicity and interpretability.
  - Used to classify the review text into positive or negative categories.
  - Hyperparameter tuning was applied to optimize the model.
  - Cross-validation was used to assess model generalization.
  
- **Random Forest** (optional comparison model):
  - Used as a secondary model to compare performance with Logistic Regression.
  - The Random Forest model tends to perform better with large and complex datasets, though it can be slower and harder to interpret.

### 4. Evaluation
- **Metrics**: 
  - Accuracy, precision, recall, and F1-score were used to evaluate model performance.
  - A confusion matrix was plotted to visualize true positives, false positives, etc.
  - ROC curves were plotted to understand model discrimination power.
- **Error Analysis**:
  - Misclassification errors were analyzed to identify areas for improvement.

### 5. Insights & Recommendations
- **Model Performance**: 
  - The logistic regression model achieved satisfactory results, and Random Forest provided a useful comparison, offering insight into the trade-offs between simplicity (Logistic Regression) and complexity (Random Forest).
  - The logistic regression model is more interpretable and faster to train, but Random Forest may provide better performance in complex, high-dimensional data.
  
## Project Setup

### Prerequisites

- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `nltk`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/logistic-regression-amazon-fine-food-reviews.git
   cd logistic-regression-amazon-fine-food-reviews
