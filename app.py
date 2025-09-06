import streamlit as st
import joblib
import re
from nltk.corpus import stopwords

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

st.title("📧 SpamClassify – Email Spam Detector")
st.write("Paste your email message below to check if it is spam or not.")

user_input = st.text_area("Enter your email message:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vect = tfidf.transform([cleaned])
    prediction = model.predict(vect)[0]
    if prediction == 1:
        st.error("🚨 Spam Detected!")
    else:
        st.success("✅ This looks safe (Ham).")
