import streamlit as st
import fitz
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import string
import unicodedata
import contractions
import nltk
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def extract_text_from_pdf(pdf_file):
    # Open the PDF file
    doc = fitz.open(pdf_file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_url(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        pdf_data = BytesIO(response.content)
        document = fitz.open(stream=pdf_data, filetype='pdf')
        text = ""
        for page in range(len(document)):
            p = document.load_page(page)
            text += p.get_text()
        return text
    return None

# Label mapping
label_mapping = {
    0: 'cable',
    1: 'fuses',
    2: 'lighting',
    3: 'others'
}

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Expand contractions
    text = contractions.fix(text)
    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Remove extra whitespace
    text = ' '.join(tokens)

    return text

model = joblib.load('naive_bayes_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_product_type(pdf_text):
    # Transform the text using the same vectorizer used during training
    X_input = vectorizer.transform([pdf_text]).toarray()
    # Predict the label
    num_prediction = model.predict(X_input)[0]   
    prediction = label_mapping[int(num_prediction)] 
    # Predict probabilities for each class
    probabilities = model.predict_proba(X_input)[0]    
     # Convert numpy.int32 to standard Python int and numpy.float64 to Python float
    class_probabilities = {label_mapping[int(model.classes_[i])]: float(probabilities[i]) for i in range(len(model.classes_))}
    return prediction, class_probabilities

st.title("PDF Product Type Classification")
st.write("Upload a PDF or provide a URL to predict the product type.")

pdf_option = st.radio("Choose input method:", ('Upload PDF', 'Enter URL'))

pdf_text = None
if pdf_option == 'Upload PDF':
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        page_text = extract_text_from_pdf(uploaded_file)
        if page_text:
            pdf_text = preprocess_text(page_text)
elif pdf_option == 'Enter URL':
    pdf_url = st.text_input("Enter PDF URL")
    if pdf_url:
        page_text = extract_text_from_url(pdf_url)
        if page_text:
            pdf_text = preprocess_text(page_text)

if pdf_text:
    prediction, probabilities = predict_product_type(pdf_text)
    st.write(f"Predicted Product Type: {prediction}")
    st.write("Class Probabilities:")
    st.write(probabilities)
