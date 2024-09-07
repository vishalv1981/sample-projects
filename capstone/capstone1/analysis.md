# Sentiment Analysis: Model Comparison Using Logistic Regression

## Project Overview
This project applies various vectorization techniques (Bag of Words, TF-IDF, Word2Vec) combined with Logistic Regression models to predict sentiment (positive or negative) from customer reviews. The analysis includes evaluation metrics such as AUC scores and regularization parameters (log(C)) to determine the best performing model.

---

## Model Comparison by AUC Score

### Analysis
The chart displays the AUC scores for four models using different vectorization techniques:
- **TF-IDF**: Achieves the highest AUC score of **0.9688**, indicating excellent performance in distinguishing between positive and negative sentiments.
- **Bag of Words**: Achieves an AUC score of **0.9398**, a reasonable performance, though slightly less effective compared to TF-IDF.
- **Average Word2Vec (W2V)**: Scores **0.9184**, showing lower performance compared to traditional techniques.
- **TF-IDF Weighted W2V**: Achieves an AUC score of **0.8971**, performing the lowest among the compared models.

### Inferences
- **TF-IDF** is the most effective vectorization technique for this sentiment analysis task, providing the best feature representation for the Logistic Regression model.
- **Word2Vec** models, while useful in other contexts, do not outperform the simpler Bag of Words and TF-IDF techniques for this dataset.

---

## Model Comparison: AUC Score and Regularization Parameter (Log(C))

### Analysis
This chart compares the AUC scores alongside the regularization parameter (log(C)) for each model:
- **Log(C)** measures the strength of regularization. A higher log(C) implies less regularization, while lower values indicate stronger regularization.
- **TF-IDF**: Achieves the highest AUC score (0.9688) with a large regularization parameter (**Log(C) = 4.00**), indicating that it benefits from less regularization.
- **Bag of Words**: Has the lowest log(C) of **-4.00** with an AUC score of 0.9398, indicating stronger regularization is applied.
- **Average W2V**: Shows balanced regularization with **Log(C) = 1.00**.
- **TF-IDF Weighted W2V**: Shows no regularization with **Log(C) = 0.00**.

### Inferences
- The **TF-IDF** model performs well with minimal regularization, suggesting that its feature-rich representation doesn't require heavy regularization to generalize well.
- **Bag of Words** benefits from more aggressive regularization, likely due to the noisier feature set from simple word counts.
- Regularization plays a crucial role in improving model performance, and fine-tuning it through hyperparameter tuning is essential for optimizing AUC.

---

## Distribution of Review Scores by Count

### Analysis
- This chart shows the distribution of review scores (0 for negative, 1 for positive), highlighting a significant **class imbalance**.
- **Positive reviews** (score = 1) dominate the dataset, with over **300,000** reviews, while **Negative reviews** (score = 0) are much fewer, with just over **50,000** reviews.

### Inferences
- **Class imbalance** is a major issue in this dataset, as positive reviews far outnumber negative reviews.
- A model trained on such an imbalanced dataset might be biased toward the majority class (positive reviews) and perform poorly on the minority class (negative reviews).
- To address this, techniques such as **resampling**, **class weighting**, or **balancing the dataset** may be necessary to improve performance on negative reviews.

---

## Distribution of Review Scores by Score

### Analysis
- This chart also displays the distribution of review scores, but with a different axis representation.
- It shows a similar trend: **positive reviews outnumber negative reviews** with over **300,000** positive reviews compared to **50,000** negative reviews.

### Inferences
- The **class imbalance problem** is again highlighted, and it needs to be addressed to ensure the model performs well on both classes.
- Without addressing this imbalance, the model might struggle to predict negative reviews accurately, leading to low recall for negative sentiment detection.

---

## Conclusion

- **TF-IDF** vectorization provides the best performance for Logistic Regression models in this sentiment analysis task.
- **Class imbalance** is a significant issue in the dataset, which must be addressed to avoid biased models.
- **Regularization** plays a crucial role in model performance, with the TF-IDF model benefiting from less regularization (higher **Log(C)**), while the **Bag of Words** model requires more regularization (lower **Log(C)**).
- **Word2Vec** models, while powerful in other contexts, do not perform as well as simpler TF-IDF or Bag of Words models for this specific sentiment analysis task.

---

#

# Analysis of Logistic Regression and Model Comparisons

## Logistic Regression with Bag of Words (BoW), TF-IDF, and Average Word2Vec

### 1. AUC vs. log(C) for Train and Cross-Validation Sets

#### Analysis:
- The AUC curve shows how the model's performance (AUC) changes with different values of the regularization parameter log(C).
  
**For Bag of Words (BoW)**:
- Both the training and cross-validation AUCs decline as log(C) increases.
- The **optimal C** value, where the cross-validation AUC stabilizes, lies around a lower value of log(C), indicating stronger regularization is beneficial.
- Increasing log(C) (reducing regularization) results in overfitting, as observed by the higher train AUC and lower cross-validation AUC.

**For TF-IDF**:
- The train and cross-validation AUCs are stable after reaching optimal log(C) values.
- TF-IDF models generalize better as they show more stable cross-validation AUC values over a wider range of log(C).

**For Average Word2Vec**:
- The curve for both train and cross-validation AUC stabilizes early and follows a similar pattern to TF-IDF, but the cross-validation AUC declines at higher log(C).
- Average Word2Vec might need lower regularization compared to BoW to avoid overfitting.

#### Inference:
- Logistic regression models using **TF-IDF** are more robust with a higher AUC performance.
- **BoW** needs more regularization to prevent overfitting.
- **Average Word2Vec** shows consistent performance but declines sharply with high log(C), indicating sensitivity to regularization.

---

### 2. Confusion Matrix for Logistic Regression Models

#### Analysis:
- **True Positive (Positive reviews correctly classified)**: 14,653
- **True Negative (Negative reviews correctly classified)**: 14,709
- **False Positive (Negative reviews incorrectly classified as positive)**: 2,449
- **False Negative (Positive reviews incorrectly classified as negative)**: 2,389

#### Inference:
- **Precision for Positive Sentiment**: The model is able to correctly predict positive sentiment at a high rate, with few misclassifications.
- **Misclassification**: There are more false positives than false negatives, indicating that the model is more prone to classifying reviews as positive.

---

### 3. ROC Curve

#### Analysis:
- The ROC curve shows the model's ability to distinguish between positive and negative reviews.
- **Train AUC = 0.9915** and **Test AUC = 0.9352**, indicating that the model performs well on the training data but shows slight overfitting when applied to test data.
- **Steep Curve**: A steep ROC curve at the beginning indicates a good separation between classes.

#### Inference:
- The model is effective at separating positive and negative reviews, with both the training and testing sets showing strong AUC performance.
- Slight overfitting could be addressed with stronger regularization or additional feature engineering.

---

### Summary:
- **Bag of Words**: Performs decently but requires stronger regularization to prevent overfitting. The AUC for cross-validation drops with increasing log(C), showing sensitivity to regularization.
- **TF-IDF**: Offers the most consistent performance across train and test datasets. The AUC remains stable even with increased regularization.
- **Average Word2Vec**: Exhibits good performance but is prone to overfitting at higher log(C) values. It performs similarly to TF-IDF but with less robustness.
- The **confusion matrix** and **ROC curve** demonstrate strong classification performance but suggest the need for more regularization to prevent overfitting.

---

### Recommendations:
- **TF-IDF** should be the preferred vectorization technique for this sentiment analysis task due to its stable performance and minimal overfitting.
- **Average Word2Vec** could be improved with fine-tuning of the regularization parameter.
- Regularization is critical, especially for **Bag of Words**, to ensure the model generalizes well on unseen data.


