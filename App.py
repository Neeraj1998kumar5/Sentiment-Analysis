import streamlit as st
import pickle
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np

# Load classical model
with open("models/classical_model.pkl", "rb") as f:
    data = pickle.load(f)
    clf = data["model"]
    tfidf = data["vectorizer"]
    le = data["label_encoder"]

# Load DistilBERT model and tokenizer
bert_model = DistilBertForSequenceClassification.from_pretrained("models/transformer_model")
tokenizer = DistilBertTokenizerFast.from_pretrained("models/transformer_model")

# Streamlit app
st.title("Sentiment Analysis App")
st.write("Analyze your text using Logistic Regression (TF-IDF) or DistilBERT.")

# Text input
user_input = st.text_area("Enter text here:", "")

# Model selection
model_choice = st.selectbox("Choose model:", ["Logistic Regression", "DistilBERT"])

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text!")
    else:
        if model_choice == "Logistic Regression":
            # Classical model prediction
            X_input = tfidf.transform([user_input])
            pred = clf.predict(X_input)[0]
            label = le.inverse_transform([pred])[0]
            st.success(f"Predicted Sentiment: {label}")

        elif model_choice == "DistilBERT":
            # DistilBERT prediction
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=64)
            outputs = bert_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            label = le.inverse_transform([pred])[0]
            st.success(f"Predicted Sentiment: {label}")
