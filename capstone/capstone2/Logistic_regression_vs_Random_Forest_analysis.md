
# Model Comparison for Text Classification

This analysis compares Logistic Regression and RandomForest models using different vectorization techniques, regularization methods, hyperparameters, and evaluates performance using the AUC metric.

## 1. Logistic Regression with Bag of Words

- **Vectorizer**: Bag of Words  
- **Regularization**: L2 (Ridge Regularization)  
- **Feature Engineering**: None  
- **Hyperparameter**: `C = 0.0001`  
- **AUC**: `0.9351`

**Pros**:
- Logistic regression with L2 regularization tends to generalize well, preventing overfitting.
- Bag of Words is a simple and interpretable technique.
- High AUC score indicates good model performance.

**Cons**:
- Bag of Words does not capture word order or semantic meaning.
- The model does not utilize any advanced feature engineering techniques.

---

## 2. RandomForest with Bag of Words

- **Vectorizer**: Bag of Words  
- **Regularization**: None  
- **Feature Engineering**: RandomizedSearchCV  
- **Hyperparameters**: 
  ```python
  {
    'n_estimators': [50, 100, 150], 
    'max_depth': [5, 10, 15], 
    'min_samples_split': [5, 10], 
    'min_samples_leaf': [1, 2, 4], 
    'bootstrap': [True, False]
  }
  ```
- **AUC**: `0.9157`

**Pros**:
- RandomForest inherently provides feature importance, aiding in interpretation.
- RandomizedSearchCV helps in selecting optimal hyperparameters.
- The model is robust and can handle a variety of data distributions.

**Cons**:
- The AUC score is slightly lower than Logistic Regression.
- RandomForest models are generally more computationally expensive than Logistic Regression.

---

## 3. Logistic Regression with TF-IDF

- **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)  
- **Regularization**: L2  
- **Feature Engineering**: None  
- **Hyperparameter**: `C = 1000`  
- **AUC**: `0.9334`

**Pros**:
- TF-IDF vectorization emphasizes important words while down-weighting common words.
- L2 regularization aids in model generalization.
- Good performance with a high AUC score.

**Cons**:
- High `C` value (1000) means less regularization, which could potentially lead to overfitting.
- No feature engineering is used, which could be explored to improve performance.

---

## 4. RandomForest with TF-IDF

- **Vectorizer**: TF-IDF  
- **Regularization**: None  
- **Feature Engineering**: RandomizedSearchCV  
- **Hyperparameters**: Same as the previous RandomForest models.  
- **AUC**: `0.9060`

**Pros**:
- TF-IDF captures the importance of terms more effectively than Bag of Words.
- RandomForest provides flexibility with hyperparameters and feature importance insights.
- Robust to overfitting due to ensemble nature.

**Cons**:
- The AUC score is lower than Logistic Regression with TF-IDF.
- RandomForest can be slower to train compared to simpler models like Logistic Regression.

---

## 5. Logistic Regression with Average Word2Vec

- **Vectorizer**: Average Word2Vec (word embeddings)  
- **Regularization**: L2  
- **Feature Engineering**: None  
- **Hyperparameter**: `C = 10`  
- **AUC**: `0.9144`

**Pros**:
- Word2Vec captures semantic meaning, which Bag of Words and TF-IDF do not.
- Logistic Regression is computationally efficient even with large vocabularies.
- Reasonably high AUC score shows good predictive performance.

**Cons**:
- Word2Vec averaging loses word order information, which could be significant for certain tasks.
- Without feature engineering, the model might not capture complex patterns.

---

## 6. RandomForest with Average Word2Vec

- **Vectorizer**: Average Word2Vec  
- **Regularization**: None  
- **Feature Engineering**: RandomizedSearchCV  
- **Hyperparameters**: Same as the previous RandomForest models.  
- **AUC**: `0.9808`

**Pros**:
- Word2Vec embeddings capture semantic and syntactic relationships.
- RandomForest with well-tuned hyperparameters (using RandomizedSearchCV) gives the highest AUC score.
- Good for complex data where relationships between features are non-linear.

**Cons**:
- Computationally expensive, especially with large hyperparameter grids and Word2Vec embeddings.
- May be prone to overfitting if the hyperparameters are not carefully tuned.

---

## Summary

### Logistic Regression
- Performs well across all vectorization techniques (Bag of Words, TF-IDF, Average Word2Vec).
- Computationally efficient and simple to interpret.
- May require more feature engineering for more complex datasets.

### RandomForest
- More computationally expensive but robust, with improved performance when using Word2Vec vectorization.
- Best AUC score achieved with Word2Vec (AUC: 0.9808).
- RandomizedSearchCV helps fine-tune hyperparameters for optimal performance.

---

## Pros and Cons of Techniques:

### Bag of Words:
- **Pros**: Simple and interpretable.
- **Cons**: Lacks semantic meaning, doesn’t consider word order.

### TF-IDF:
- **Pros**: Captures term importance and reduces the impact of common words.
- **Cons**: Still lacks semantic relationships between words.

### Average Word2Vec:
- **Pros**: Captures semantic and syntactic information.
- **Cons**: Loses sentence structure information, which may affect performance in certain tasks.

---

## Overall Recommendations:

- For simpler and faster models, **Logistic Regression** with **TF-IDF** is a strong candidate, balancing performance and efficiency.
- For more complex datasets, **RandomForest** with **Word2Vec** is highly recommended due to its ability to capture semantic information and relationships between features, resulting in the highest AUC score in this analysis.
