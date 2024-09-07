# Analysis and Inferences

## 1. Best Performing Model (Highest AUC)
- **Bag of Words with C=0.0001** achieves the highest AUC score of **0.9352**.

### Inference:
- **Bag of Words**, despite being one of the simpler vectorization methods, works effectively for this dataset. The higher AUC suggests that the model can distinguish between positive and negative reviews with good accuracy. 
- This could be due to the nature of the text data, where simple frequency-based representations are sufficient to capture sentiment-related patterns.

---

## 2. Close Performance of TF-IDF
- **TF-IDF with C=10000** achieves an AUC score of **0.9335**, which is very close to the Bag of Words model.

### Inference:
- **TF-IDF** captures both term frequency and inverse document frequency, giving more weight to informative words that are rare in the dataset. 
- While it performs similarly to Bag of Words, it does not significantly outperform it. This suggests that the dataset may not have many rare but informative words that could have leveraged TF-IDF’s strength.

---

## 3. Average Word2Vec Model Performance
- **Average Word2Vec with C=10** achieves a slightly lower AUC score of **0.9137** compared to Bag of Words and TF-IDF.

### Inference:
- **Word2Vec** creates dense, continuous vector representations of words that capture semantic meaning. 
- The slightly lower AUC could suggest that the semantic information captured by Word2Vec is not as useful for this particular dataset, which might benefit more from the direct frequency-based approaches of Bag of Words or TF-IDF. 
- Additionally, averaging Word2Vec embeddings might lose contextual information, leading to slightly lower performance.

---

## 4. Effect of Regularization Parameter (C)
- The regularization parameter **C** varies across the models. For the best-performing Bag of Words model, a small C (**0.0001**) performs best, indicating stronger regularization. 
- For TF-IDF and Word2Vec, larger values of C (**10000** and **10**, respectively) were optimal, suggesting that less regularization is needed for these vectorizations.

### Inference:
- The regularization strength (controlled by **C**) is an important hyperparameter. 
- **Bag of Words**, being a simpler model, benefits from stronger regularization to avoid overfitting, while **TF-IDF** and **Word2Vec** require less regularization for optimal performance.

---

## General Observations
- All models perform reasonably well, with **AUC scores above 0.91**, indicating that the Logistic Regression models are effective at predicting sentiment.
- The **Bag of Words** approach, while simple, performs the best, highlighting that sometimes simpler methods can outperform more complex approaches (like Word2Vec) in specific contexts.

---

## Recommendations
- **Consider Bag of Words** as a baseline model for this dataset, given its simplicity and strong performance.
- For more sophisticated models, **TF-IDF** could be a good alternative, as it provides similar results and may perform better if more rare but important terms are present in the data.
- **Word2Vec** could be improved by using more advanced techniques like incorporating contextual word embeddings (e.g., using models like BERT) to better capture the semantic context of words, potentially improving performance.
- Experimenting with different **feature engineering techniques** (e.g., combining different vectorization techniques) could further improve model performance.
