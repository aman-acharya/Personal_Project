from pipe import TextPreprocessor, TextTokenizer
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

# Load the models
Logistic = joblib.load('logistic_pipeline.pkl')
SVM = joblib.load('svm_pipeline.pkl')
RF = joblib.load('rf_pipeline.pkl')
NB = joblib.load('nb_pipeline.pkl')

models = {'Logistic Regression': Logistic, 'Support Vector Machine': SVM, 'Random Forest': RF, 'Naive Bayes': NB}

# Create a title for the app
st.title('Sentiment Analysis App')

st.sidebar(title='Sentiment Analysis App')
st.sidebar.write('This is a simple sentiment analysis app that uses four different models to predict the sentiment of a given text. The models are trained on the IMDB dataset and can predict whether a given text is positive or negative. The models are Logistic Regression, Support Vector Machine, Random Forest, and Naive Bayes.')
st.sidebar.write('Please enter a text in the text box below and click the "Predict" button to see the predictions of the models.')

# give the user to select which model to use in the sidebar
model = st.sidebar.selectbox('Select a model', list(models.keys()))

# Create a text box for the user to enter the text
text = st.text_area('Enter the text here:')

# Create a button to predict the sentiment
if st.button('Predict'):
    if text:
        # Load the text preprocessor and tokenizer
        preprocessor = TextPreprocessor()
        tokenizer = TextTokenizer()

        # Preprocess the text
        text = preprocessor.transform([text])

        # Tokenize the text
        text = tokenizer.transform(text)

        # Predict the sentiment
        prediction = models[model].predict(text)

        # Display the prediction
        st.write(f'The sentiment of the text is: {prediction[0]}')
    else:
        st.write('Please enter a text in the text box above to predict the sentiment.')






