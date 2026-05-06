import streamlit as st
from classifier import train_classifier
from sentiment import analyze_sentiment
from preprocess import clean_text

@st.cache_resource
def load_model():
    return train_classifier("data/News_Category_Dataset_v3.json")

MODEL = load_model()

st.title("📰 News Topic & Sentiment Analyzer")
st.write("Enter a news headline or description below:")

user_input = st.text_area("News text:")

if st.button("Analyze") and user_input.strip():
    cleaned = clean_text(user_input)
    category = MODEL.predict([cleaned])[0]
    sentiment = analyze_sentiment(user_input)

    st.write(f"**Predicted Category:** {category}")
    st.write(f"**Sentiment Tone:** {sentiment}")
