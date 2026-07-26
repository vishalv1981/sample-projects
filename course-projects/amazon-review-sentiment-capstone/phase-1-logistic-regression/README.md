# Sentiment Analysis with Multiple Logistic Regression Models

## Project Overview
This project implements various Logistic Regression models along with RandomForest for sentiment analysis on customer reviews. The dataset is preprocessed, cleaned, and vectorized using techniques such as Bag of Words (BoW), TF-IDF, and Word2Vec to extract features. These features are then fed into the models to predict sentiment.

## Key Features:
- Logistic Regression and RandomForest Classifier
- Evaluation Metrics:
  - AUC (Area Under the Curve)
  - ROC (Receiver Operating Characteristic) Curve
  - Confusion Matrix
  - Misclassification Error

## Vectorization Techniques:
- Bag of Words (BoW)
- TF-IDF
- Word2Vec
- TF-IDF Weighted Word2Vec

## Data Preprocessing, Cleaning, and Text Processing

### Data Preprocessing and Cleanup
The dataset was sourced from an SQLite database containing customer reviews. Several cleaning and preprocessing steps were applied before the model training and evaluation process.

#### Steps for Data Cleanup:

1. **Filtering Reviews:**
   - Only reviews with a score different from 3 were selected for analysis. Reviews with scores greater than 3 were labeled as "positive" (1), while reviews with scores less than 3 were labeled as "negative" (0).
   - Neutral reviews (score = 3) were excluded to prevent ambiguity.

2. **Data Deduplication:**
   - Duplicate reviews were removed based on multiple fields such as UserId, ProfileName, Time, and Text.

3. **Helpfulness Metric Validation:**
   - Reviews where the HelpfulnessNumerator exceeded HelpfulnessDenominator were removed to ensure valid data.

4. **Dataset Size After Cleanup:**
   - After filtering, deduplication, and validation, the dataset contained a balanced set of positive and negative reviews.

5. **Data Visualization:**
   - Seaborn count plots were used to visualize the distribution of positive and negative reviews in the cleaned dataset.

### Text Processing
To ensure high-quality textual features, various text preprocessing techniques were applied to clean and standardize the review text:

1. **HTML Tag Removal:**
   - HTML tags were removed using regex and BeautifulSoup.

2. **Stopwords Removal:**
   - A customized list of stopwords was created using the NLTK stopwords corpus. Negations (e.g., "not", "don't") were retained, while other stopwords were removed to reduce noise.

3. **Stemming:**
   - A Snowball Stemmer was used to reduce words to their base forms (e.g., "tasty" -> "tasti").

4. **Contraction Expansion:**
   - Contractions (e.g., "won't" -> "will not") were expanded for better accuracy during model training.

5. **Punctuation and Special Character Removal:**
   - Punctuation and special characters were removed using regular expressions, leaving only alphanumeric characters.

6. **Handling URLs:**
   - URLs were removed using regex patterns.

7. **Preprocessing Function:**
   - A unified `preprocess_sentence` function was created to perform all the steps above.

8. **Applying the Preprocessing Function:**
   - The `preprocess_sentence` function was applied to the review text and summary columns to create cleaned versions for modeling.

9. **Time Conversion:**
   - The timestamp was converted to a readable datetime format using `pd.to_datetime` for consistency.

## Feature Engineering
Different vectorization techniques were used to convert the processed text data into numerical features:
- Bag of Words (BoW)
- TF-IDF
- Word2Vec
- TF-IDF Weighted Word2Vec

These vectorized features were fed into the Logistic Regression models for training.

## Model Training and Evaluation
Multiple Logistic Regression models were trained with different regularization strengths. The project includes detailed evaluation metrics such as:
- AUC
- ROC curves
- Confusion matrix
- Misclassification error

Additionally, a RandomForest Classifier was included as a comparative model with hyperparameter tuning using `GridSearchCV`.

## Results
The results were evaluated based on the following metrics:
- **AUC Score:** Provides insight into the model's ability to differentiate between positive and negative sentiments.
- **ROC Curve:** Visualizes the performance of the classification models at different threshold settings.
- **Confusion Matrix:** Displays true positives, true negatives, false positives, and false negatives.
- **Misclassification Error:** Measures the rate of incorrect predictions.
