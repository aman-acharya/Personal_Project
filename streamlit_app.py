from pipe import TextPreprocessor, TextTokenizer
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from wordcloud import WordCloud
# Load the models
Logistic = joblib.load('logistic_pipeline.pkl')
SVM = joblib.load('svm_pipeline.pkl')
RF = joblib.load('rf_pipeline.pkl')
NB = joblib.load('nb_pipeline.pkl')

models = {'Logistic Regression': Logistic, 'Support Vector Machine': SVM, 'Random Forest': RF, 'Naive Bayes': NB}

# Create a title for the app
st.title('Sentiment Analysis App')

st.sidebar.write(title='Sentiment Analysis App')
st.sidebar.write('This is a simple sentiment analysis app that uses four different models to predict the sentiment of a given text. The models are trained on the review of a restaurant in US and can predict whether a given review/text is positive or negative. The models are Logistic Regression, Support Vector Machine, Random Forest, and Naive Bayes.')
st.sidebar.write('Please enter a text in the text box below and click the "Predict" button to see the predictions of the models.')

# give the user to select which model to use in the sidebar
model = st.sidebar.selectbox('Select a model', list(models.keys()))

# Create a text box for the user to enter the text
text = st.text_area('Enter the text here:')


# Create a button to predict the sentiment
if st.button('Predict'):
    model = models[model]
    prediction = model.predict([text])
    prediction = 'Positive' if prediction[0] == 1 else 'Negative'
    st.write(f'The predicted sentiment is: {prediction}')

# add a word cloud generator
st.sidebar.write('Word Cloud Generator')
st.sidebar.write('This is a simple word cloud generator that generates a word cloud from the given text. The word cloud shows the most common words in the text.')
st.sidebar.write('Please enter a text in the text box below and click the "Generate" button to see the word cloud.')

if st.sidebar.button('Generate'):
    text = TextPreprocessor().transform(text)
    text = TextTokenizer().transform(text)
    text = ' '.join(text)
    st.write('The word cloud is generated below:')
    word_cloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    st.image(word_cloud.to_array())


        




