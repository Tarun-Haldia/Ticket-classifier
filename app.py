import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import gradio as gr

# NLTK downloads
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset (ensure this file is present or replace with your dataset path)
df = pd.read_excel("ai_dev_assignment_tickets_complex_1000.xls", engine="xlrd")

# ---- Preprocessing ----
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t, pos='v') for t in tokens]
    return ' '.join(tokens)

df['ticket_text'] = df['ticket_text'].astype(str)
df['clean_text'] = df['ticket_text'].apply(preprocess_text)

# Features
vectorizer = TfidfVectorizer(max_features=500)
X_tfidf = vectorizer.fit_transform(df['clean_text'])
df['ticket_length'] = df['clean_text'].apply(lambda x: len(x.split()))
df['sentiment'] = df['ticket_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

X = np.hstack([X_tfidf.toarray(), df[['ticket_length', 'sentiment']].values])
df = df.dropna(subset=['issue_type', 'urgency_level'])

y_issue = df['issue_type']
y_urgency = df['urgency_level']
X_cleaned = X[df.index]

# Train/test split
X_train, X_test, y_issue_train, y_issue_test, y_urgency_train, y_urgency_test = train_test_split(
    X_cleaned, y_issue, y_urgency, test_size=0.2, random_state=42
)

# Models
issue_model = LogisticRegression(max_iter=1000)
issue_model.fit(X_train, y_issue_train)

urgency_model = LogisticRegression(max_iter=1000)
urgency_model.fit(X_train, y_urgency_train)

# Product + keyword extractors
product_list = df['product'].dropna().astype(str).unique().tolist()
complaint_keywords = ['broken', 'not working', 'damaged', 'late', 'error', 'crash', 'delayed', 'failure', 'installation', 'lost']

def normalize_text(txt): return re.sub(r'[^a-z0-9]', '', txt.lower())
def extract_product(text):
    return [prod for prod in product_list if normalize_text(prod) in normalize_text(text)]
def extract_keywords(text):
    return [word for word in complaint_keywords if word in text.lower()]
def extract_dates(text):
    date_patterns = [
        r'\d{1,2}\s(?:January|February|...|December)\s\d{4}',
        r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
        r'(?:Jan|Feb|...|Dec)\s\d{1,2},?\s\d{4}',
        r'\d{1,2}\s(?:Jan|Feb|...|Dec)\s\d{4}'
    ]
    found_dates = []
    for pattern in date_patterns:
        found_dates.extend(re.findall(pattern, text, re.IGNORECASE))
    return found_dates

def extract_entities(text):
    return {
        "product_names": extract_product(text),
        "dates": extract_dates(text),
        "keywords": extract_keywords(text)
    }

# Prediction pipeline
def process_ticket(text):
    cleaned = preprocess_text(text)
    tfidf_vector = vectorizer.transform([cleaned])
    length = len(cleaned.split())
    sentiment = TextBlob(text).sentiment.polarity
    combined_features = np.hstack([tfidf_vector.toarray(), [[length, sentiment]]])
    issue_pred = issue_model.predict(combined_features)[0]
    urgency_pred = urgency_model.predict(combined_features)[0]
    entities = extract_entities(text)
    return issue_pred, urgency_pred, entities

# Gradio app
def gradio_interface(text):
    issue, urgency, entities = process_ticket(text)
    return issue, urgency, entities

interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(lines=5, label="Enter Customer Support Ticket"),
    outputs=[
        gr.Text(label="Predicted Issue Type"),
        gr.Text(label="Predicted Urgency Level"),
        gr.JSON(label="Extracted Entities")
    ],
    title="Ticket Classifier & Entity Extractor",
    description="Paste any support ticket to see predictions and extracted entities."
)

import os

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))  # Render PORT env var, fallback 10000
    interface.launch(server_name="0.0.0.0", server_port=port)
