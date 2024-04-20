import streamlit as st
if __name__ == '__main__':

    st.title("Sentiment Analysis App")
    # rest of Streamlit code
    st.write("Model Performance")
    st.write(f"Naive Bayes Cross Validation Score: {scores.mean()}") 
    st.write("Explore the Data")
    st.write(data.head())