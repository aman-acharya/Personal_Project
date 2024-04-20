import pandas as pd
import numpy as np
import spacy
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tokenizer=word_tokenize):
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = []
        for text in X:
            text = text.lower()  # Convert to lowercase
            text = re.sub('[^a-zA-Z]', ' ', text)  # Keep only lowercase and uppercase letters
            text = re.sub('\s+', ' ', text)  # Replace multiple whitespace characters with a single space
            text = self.tokenizer(text)  # Tokenize the text
            X_transformed.append(text)
        return X_transformed

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
    
class SentimentPipeline:
    def __init__(self, classifier=SentimentClassifier()):
        self.classifier = classifier

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        return self.classifier.predict(X)

    def score(self, X, y):
        return self.classifier.score(X, y)