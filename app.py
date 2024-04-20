import streamlit as st
from pipeline import TextPreprocessor, TextVectorizer, SentimentClassifier
import joblib

def load_pipeline(model_path):
    pipeline = joblib.load(model_path)
    return pipeline

def main():
    st.title("Sentiment Analysis")
    text = st.text_input("Enter text:")
    if text:
        pipeline = load_pipeline("Nlp_pipeline.pkl")
        result = pipeline.predict([text])
        st.write(f"Sentiment: {result[0]}")

if __name__ == "__main__":
    main()
