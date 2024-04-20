import pandas as pd
import numpy as np
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
# load the data
data = pd.read_csv('restaurant_reviews.csv')
data.head()

nlp = spacy.load('en_core_web_sm')
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)  # Keep only lowercase and uppercase letters
    text = re.sub('\s+', ' ', text)  # Replace multiple whitespace characters with a single space
    text = nlp(text)
    text = [word.lemma_ for word in text if not word.is_stop and not word.is_punct and not word.like_num]
    text = ' '.join(text)
    return text

# Define a custom tokenizer
def custom_tokenizer(text):
    # Use NLTK's word_tokenize function to tokenize the text
    tokens = word_tokenize(text)
    # Return the tokens
    return tokens

# lets create a new feature 'Length' and 'Sentiment'
data['Length'] = data['Review'].apply(len)
data['Sentiment'] = data['Rating'].apply(lambda x: 'Positive' if x >= 4 else 'Negative')

# lets extract equal number of positive and negative reviews
positive_reviews = data[data['Sentiment'] == 'Positive'].sample(2238)
negative_reviews = data[data['Sentiment'] == 'Negative'].sample(2238)

data = pd.concat([positive_reviews, negative_reviews]).reset_index(drop=True)
data['Review'] = data['Review'].apply(clean_text)

# Convert text data into numerical features using TF-IDF
tfidf = TfidfVectorizer(tokenizer=custom_tokenizer)

X = tfidf.fit_transform(data['Review'])
y = data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# lets try naive bayes 
model = MultinomialNB(alpha=2)
model.fit(X_train, y_train)
scores = cross_val_score(model, X_train, y_train, cv=5)
print('Cross-validation scores: {}'.format(scores))
print('Average cross-validation score: {:.2f}'.format(scores.mean()))

model = SVC(C=10, gamma=1, kernel='rbf')
model.fit(X_train, y_train)
scores = cross_val_score(model, X_train, y_train, cv=5)
print('Cross-validation scores: {}'.format(scores))
print('Average cross-validation score: {:.2f}'.format(scores.mean()))