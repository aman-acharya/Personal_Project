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

# Create a title for the app
st.title('Sentiment Analysis App')

# lets give user option to chose the model for prediction
model = st.selectbox('Select the model', ('Logistic Regression', 'SVM', 'Random Forest', 'Naive Bayes'))

# Get the review from the user
review = st.text_area('Enter your review here')

# Make the prediction
if st.button('Predict'):
    if model == 'Logistic Regression':
        prediction = Logistic.predict([review])
    elif model == 'SVM':
        prediction = SVM.predict([review])
    elif model == 'Random Forest':
        prediction = RF.predict([review])
    else:
        prediction = NB.predict([review])
    
    st.write(f'The review is: {prediction[0]}')

# Add a footer
st.markdown('Built with Streamlit by [Aman Acharya]')




