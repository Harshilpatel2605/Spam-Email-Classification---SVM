import sys
import os

# Ensure we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from src.predict import predict_spam

st.set_page_config(page_title="Spam Detector", page_icon="📩", layout="centered")

st.title("📩 Spam Detector (Custom SVM + TF-IDF)")

msg = st.text_area("Enter your message:")

if st.button("Check"):
    result = predict_spam(msg)
    st.success("✅ Not Spam" if result == 0 else "🚫 Spam")
