Sentiment Analysis with Multiple Logistic Regression Models
Project Overview
This project implements various Logistic Regression models along with RandomForest for sentiment analysis on customer reviews. The dataset is preprocessed, cleaned, and vectorized using techniques such as Bag of Words (BoW), TF-IDF, and Word2Vec to extract features. These features are then fed into the models to predict sentiment.

Key Features:
Logistic Regression and RandomForest Classifier
Evaluation Metrics:
AUC (Area Under the Curve)
ROC (Receiver Operating Characteristic) Curve
Confusion Matrix
Misclassification Error
Vectorization Techniques:
Bag of Words (BoW)
TF-IDF
Word2Vec
TF-IDF Weighted Word2Vec
Data Preprocessing, Cleaning, and Text Processing
Data Preprocessing and Cleanup
The dataset was sourced from an SQLite database containing customer reviews. The dataset underwent several cleaning and preprocessing steps before being used in the model training and evaluation process.

Steps for Data Cleanup:
Filtering Reviews:

Only reviews with a score different from 3 were selected for analysis. Reviews with scores greater than 3 were labeled as "positive" (1), while reviews with scores less than 3 were labeled as "negative" (0).
Neutral reviews (score = 3) were excluded to prevent ambiguity.
Data Deduplication:

Duplicate reviews were removed based on multiple fields such as UserId, ProfileName, Time, and Text.
Helpfulness Metric Validation:

Reviews where HelpfulnessNumerator exceeded HelpfulnessDenominator were removed to ensure valid data.
Dataset Size After Cleanup:

After filtering, deduplication, and validation, the dataset contained a balanced set of positive and negative reviews.
Data Visualization:

Seaborn count plots were used to visualize the distribution of positive and negative reviews in the cleaned dataset.
Text Processing
To ensure high-quality textual features, various text preprocessing techniques were applied to clean and standardize the review text:

1. HTML Tag Removal
The text data often contained HTML tags that needed to be removed. Using regex and BeautifulSoup, reviews containing HTML tags were identified and cleaned.

2. Stopwords Removal
A customized list of stopwords was created based on the NLTK stopwords corpus. Certain stopwords such as negations ('not', 'don't', etc.) were retained to preserve sentiment context, while others were removed to reduce noise in the data.

3. Stemming
A Snowball Stemmer was initialized to stem words in the reviews, reducing words to their base forms. For example, 'tasty' becomes 'tasti'.

4. Contraction Expansion
The reviews often contained contractions like 'won't', 'can't', etc., which were expanded to their full forms (e.g., 'won't' -> 'will not'). This step ensured better accuracy during model training by avoiding confusion between different forms of words.

5. Punctuation and Special Character Removal
Punctuation and special characters were removed using regular expressions, keeping only alphanumeric characters.

6. Handling URLs
Reviews often contained URLs, which were irrelevant to sentiment analysis. These URLs were removed using regex patterns to clean the text.

7. Preprocessing Function
A unified preprocess_sentence function was created to perform all the steps mentioned above. This function removes URLs, expands contractions, cleans HTML tags, removes stopwords, and normalizes the text for model training.

8. Applying the Preprocessing Function
The preprocess_sentence function was applied to the review text and summary columns, creating new cleaned versions of these fields in the dataset for use in modeling.

9. Time Conversion
The timestamp in the dataset was converted to a readable datetime format, using pd.to_datetime with a time unit of seconds, for consistency.

Feature Engineering
The project uses different vectorization techniques to convert the processed text data into numerical features:

Bag of Words (BoW)
TF-IDF
Word2Vec
TF-IDF Weighted Word2Vec
These vectorized features are fed into the Logistic Regression models for training.

Model Training and Evaluation
Multiple Logistic Regression models are trained with different regularization strengths. The project includes detailed evaluation metrics such as AUC, ROC curves, confusion matrix, and misclassification error.

Additionally, a RandomForest Classifier is included as a comparative model, with hyperparameter tuning using GridSearchCV.

Results
The results are evaluated based on:

AUC Score: Provides insight into the model's ability to differentiate between positive and negative sentiments.
ROC Curve: Visualizes the performance of the classification models at different threshold settings.
Confusion Matrix: Displays true positives, true negatives, false positives, and false negatives.
Misclassification Error: Measures the rate of incorrect predictions.
