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

## Future Work
- Implement class balancing techniques to address the class imbalance issue.
- Experiment with other vectorization techniques or deep learning models to further improve performance.
- Explore additional regularization techniques and hyperparameter tuning to fine-tune model performance.

