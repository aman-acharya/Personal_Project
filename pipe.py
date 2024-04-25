from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
nltk.download('stopwords')
nltk.download('wordnet')
# nltk.download('punkt')


# lets crteate a class to preprocess the review
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stopwords=True, remove_punctuations=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.remove_punctuations = remove_punctuations
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            X = X.apply(self._clean_text)
        elif isinstance(X, list):
            X = [self._clean_text(text) for text in X]
        return X
    
    def _clean_text(self, text):
        text = text.lower()
        if self.remove_punctuations:
            text = self._remove_punctuations(text)
        if self.remove_stopwords:
            text = self._remove_stopwords(text)
        if self.lemmatize:
            text = self._lemmatize(text)
        return text
    
    def _remove_punctuations(self, text):
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        return text
    
    def _remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words])
        return text
    
    def _lemmatize(self, text):
        text = ' '.join([self.lemmatizer.lemmatize(word) for word in text.split()])
        return text
    
class TextTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = TfidfVectorizer()
        
    def fit(self, X, y=None):
        self.tokenizer.fit(X)
        return self
    
    def transform(self, X):
        return self.tokenizer.transform(X)

Pipeline([('preprocessor', TextPreprocessor()), ('tokenizer', TextTokenizer()), ('model', LogisticRegression())])
# Create a pipeline for the logistic regression model
logistic_pipeline = Pipeline([
    ('preprocessor', TextPreprocessor()),
    ('tokenizer', TextTokenizer()),
    ('model', LogisticRegression())
])

