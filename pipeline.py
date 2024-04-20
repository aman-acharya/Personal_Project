import pandas as pd
import numpy as np
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
from sklearn.feature_extraction.X import TfidfVectorizer
from nltk.tokenize import word_tokenize
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import nlp


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer=word_tokenize):
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.lower()
        X = re.sub('[^a-zA-Z]', ' ', X)  # Keep only lowercase and uppercase letters
        X = re.sub('\s+', ' ', X)  # Replace multiple whitespace characters with a single space
        X = nlp(X)
        X = [word.lemma_ for word in X if not word.is_stop and not word.is_punct and not word.like_num]
        X = ' '.join(X)
        return X

class TextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer=TfidfVectorizer):
        self.vectorizer = vectorizer

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X):
        X_transformed = self.vectorizer.transform(X)
        return X_transformed
    
class SentimentClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier=LogisticRegression):
        self.classifier = classifier

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X):
        return self.classifier.predict(X)

    def score(self, X, y):
        return self.classifier.score(X, y)
    
