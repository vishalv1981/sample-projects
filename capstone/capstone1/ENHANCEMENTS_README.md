# Scope of capstone1
The project has implemented the LR for the sentiment analysis. As part of the Capstone2 will extend to **Random Forest model** and also would try the **Vader model for sentiment analysis**
The project uses Stemming and has few drawbacks. will implement the **Lemmatization for sentiment analysis**



# Impact of Lemmatization on Sentiment Analysis Models

## Introduction
The introduction of lemmatization in the text preprocessing pipeline will significantly impact the performance and behavior of sentiment analysis models. This README outlines how the change from stemming to lemmatization affects various aspects of the models.

## Key Benefits of Lemmatization

### 1. Improved Word Standardization and Meaning Preservation
- **With Stemming**: Words are often reduced to non-standard forms (e.g., "tasty" → "tasti"). This can cause the model to treat similar words differently, reducing its ability to generalize.
- **With Lemmatization**: Words are converted to their dictionary forms (e.g., "better" → "good", "tasted" → "taste"). This maintains the semantic meaning of words and results in more meaningful features for the model.
- **Impact**: Lemmatization helps the model treat words with similar meanings as the same, improving model generalization and reducing noise.

### 2. More Accurate Feature Representation
- **With Stemming**: The vocabulary may include non-standard or meaningless words, potentially leading the model to learn irrelevant patterns.
- **With Lemmatization**: The vocabulary will consist of meaningful words (e.g., "run", "running", and "ran" all become "run"), leading to more accurate feature representation.
- **Impact**: Better feature representation results in improved model accuracy when using techniques like Bag of Words or TF-IDF.

### 3. Handling Different Word Forms
- **With Stemming**: Different word forms might be treated inconsistently (e.g., "running" and "runs" might not be stemmed to the same root).
- **With Lemmatization**: All forms of a word are mapped to a base form (e.g., "run" for all verb tenses), providing more consistent feature extraction.
- **Impact**: More consistent word handling improves model learning and generalization to unseen data.

### 4. Reduction in Vocabulary Size
- **With Stemming**: While the vocabulary size is reduced, it may include incorrect forms.
- **With Lemmatization**: The vocabulary is larger but consists of meaningful words, leading to better predictions.
- **Impact**: Though the vocabulary might slightly increase, the features are more refined, leading to better model performance.

### 5. Enhanced Model Interpretability
- **With Stemming**: It can be challenging to interpret the model due to non-standard word forms (e.g., "tasti" instead of "tasty").
- **With Lemmatization**: Dictionary words make model outputs easier to interpret, analyze feature importance, and explain predictions.
- **Impact**: Improved interpretability makes it easier to understand and refine the model.

### 6. Better Model Generalization
- **With Stemming**: Over-aggressive reduction of words can cause the model to overfit on specific forms.
- **With Lemmatization**: Lemmatization ensures consistency across different word forms, which helps the model generalize better.
- **Impact**: Improved generalization leads to better performance on unseen data.

### 7. Impact on Model Performance
- **With Stemming**: Loss of meaningful word distinctions may reduce model performance (accuracy, precision, recall, AUC).
- **With Lemmatization**: By keeping words in their correct forms, the model is likely to perform better across multiple metrics.
- **Impact**: Expect improvements in model accuracy, precision, recall, and AUC due to cleaner and more meaningful input data.

### 8. Computational Overhead
- **With Stemming**: Computationally faster due to simple truncation rules.
- **With Lemmatization**: More computationally intensive as it requires dictionary lookups and POS tagging.
- **Impact**: Although preprocessing time increases, the improvements in model performance and accuracy outweigh the added cost.

## Overall Impact Summary:
- **Accuracy**: Likely to improve due to more meaningful features and reduced noise.
- **Model Generalization**: Enhanced by treating word variants consistently.
- **Vocabulary Size**: More manageable and consistent, improving vectorization efficiency.
- **Interpretability**: Easier to understand the model's decision-making process.
- **Processing Time**: Lemmatization takes more time than stemming, but the performance gain justifies the additional time for most use cases.

## Testing the Impact:
To assess the effect of lemmatization on your models, compare the following performance metrics before and after the change:
- Accuracy
- Precision and Recall
- AUC (Area Under the ROC Curve)
- F1 Score

