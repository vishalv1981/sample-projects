import re
import sqlite3
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
import swifter  # Ensure swifter is installed for faster row-wise operations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
from prettytable import PrettyTable
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    """
    Convert TreeBank POS tags to WordNet POS tags for lemmatization.
    
    :param treebank_tag: POS tag from NLTK's pos_tag
    :return: Corresponding WordNet POS tag
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if no match

# Establish connection to the SQLite database
con = sqlite3.connect('/path_to_your_database/finefoodreviews/database.sqlite')

# Filtering only Positive and Negative Reviews (Score != 3)
filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score!=3""", con)

# Partitioning scores into Positive (1) and Negative (0)
def partition(x):
    return 0 if x < 3 else 1

filtered_data['Score'] = filtered_data['Score'].map(partition)
print("No. of data points in Dataset:", filtered_data.shape)

# Deduplication based on UserId, ProfileName, Time, and Text
final_data = filtered_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, keep='first', inplace=False)
print("Shape after deduplication:", final_data.shape)

# Remove entries where HelpfulnessNumerator > HelpfulnessDenominator
final_data = final_data[final_data['HelpfulnessNumerator'] <= final_data['HelpfulnessDenominator']]
print("Shape after Helpfulness validation:", final_data.shape)

# Convert 'Time' from UNIX timestamp to datetime
final_data['Time'] = pd.to_datetime(final_data['Time'], unit='s')
print(final_data.head())

# --- Text Processing Section with Lemmatization ---

# Initialize stopwords and customize the list
stop_words = set(stopwords.words('english'))  # Create a set of stopwords from NLTK

# List of stopwords to exclude (retain these in the customized stopwords)
excluding_stop = [
    'against', 'not', 'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
    'haven', "haven't", 'isn', "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 
    'needn', "needn't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 
    'won', "won't", 'wouldn', "wouldn't"
]

# Remove excluded stopwords from the stop_words set
stop_words = stop_words - set(excluding_stop)

# Initialize the WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function to expand contractions in the text
def decontracted(phrase):
    phrase = re.sub(r"won't", 'will not', phrase)
    phrase = re.sub(r"can\'t", 'can not', phrase)
    phrase = re.sub(r"n\'t", ' not', phrase)
    phrase = re.sub(r"\'re", ' are', phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Function to preprocess sentences with lemmatization
def preprocess_sentence(sentence):
    """
    Preprocesses a given sentence by performing the following steps:
    1. Remove URLs.
    2. Remove HTML tags.
    3. Expand contractions.
    4. Remove words containing digits.
    5. Remove special characters.
    6. Convert to lowercase, remove stopwords, and lemmatize.
    
    :param sentence: The input sentence to preprocess.
    :return: The preprocessed sentence.
    """
    # Remove URLs
    sentence = re.sub(r"http\S+", "", sentence)
    
    # Remove HTML tags
    sentence = BeautifulSoup(sentence, "html.parser").get_text()
    
    # Expand contractions
    sentence = decontracted(sentence)
    
    # Remove words containing digits
    sentence = re.sub(r"\S*\d\S*", "", sentence).strip()
    
    # Remove special characters, keeping only letters
    sentence = re.sub(r'[^A-Za-z]+', ' ', sentence)
    
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    
    # POS tagging
    pos_tags = pos_tag(tokens)
    
    # Lemmatize each token with appropriate POS tag
    lemmatized_tokens = [
        lemmatizer.lemmatize(token.lower(), get_wordnet_pos(pos)) 
        for token, pos in pos_tags 
        if token.lower() not in stop_words
    ]
    
    # Join tokens back into a single string
    cleaned_sentence = ' '.join(lemmatized_tokens)
    
    return cleaned_sentence.strip()

# Apply the preprocessing function to the 'Text' column
print("Starting text preprocessing with lemmatization...")
final_data['Cleaned_Text'] = final_data['Text'].swifter.apply(preprocess_sentence)

# Logistic Regression Model Training

# Taking equal number of positive and negative data points for balanced dataset
data_pos = final_data[final_data['Score'] == 1].sample(n=57000, random_state=42)
data_neg = final_data[final_data['Score'] == 0].sample(n=57000, random_state=42)
final_balanced = pd.concat([data_pos, data_neg])
print("Balanced dataset shape:", final_balanced.shape)

# Feature matrix (X) and target (y)
X = final_balanced['Cleaned_Text']
y = final_balanced['Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorization using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vect.fit_transform(X_train)
X_test_tfidf = tfidf_vect.transform(X_test)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(X_train_tfidf, y_train)

# Model Evaluation
y_train_pred_prob = log_reg.predict_proba(X_train_tfidf)[:, 1]
y_test_pred_prob = log_reg.predict_proba(X_test_tfidf)[:, 1]

# ROC AUC Score
auc_train = roc_auc_score(y_train, y_train_pred_prob)
auc_test = roc_auc_score(y_test, y_test_pred_prob)
print(f"Train AUC: {auc_train:.4f}")
print(f"Test AUC: {auc_test:.4f}")

# Plot ROC Curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_prob)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_prob)

plt.figure(figsize=(10, 6))
plt.plot(fpr_train, tpr_train, label=f'Train ROC (AUC = {auc_train:.4f})', color='green')
plt.plot(fpr_test, tpr_test, label=f'Test ROC (AUC = {auc_test:.4f})', color='blue')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Confusion Matrix
y_test_pred = log_reg.predict(X_test_tfidf)
confusion_mat = confusion_matrix(y_test, y_test_pred)
class_labels = ["Negative", "Positive"]
df_confusion_mat = pd.DataFrame(confusion_mat, columns=class_labels, index=class_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(df_confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5, linecolor='black')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Misclassification Error
misclassification_error = 1 - accuracy_score(y_test, y_test_pred)
print(f"Misclassification Error: {misclassification_error:.4f}")

# Save final data to SQLite for future use
connection = sqlite3.connect('final12.sqlite')
final_data.to_sql('Reviews', connection, if_exists='replace', index=True)
print("Final data stored in 'final12.sqlite' database.")
connection.close()
