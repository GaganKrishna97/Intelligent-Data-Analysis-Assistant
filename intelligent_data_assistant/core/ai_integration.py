from transformers import pipeline
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
import streamlit as st
import pandas as pd
from math import sqrt

# Initialize HF pipelines (can be slow to load first time)
try:
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except Exception:
    sentiment_pipe = None

try:
    summarize_pipe = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
except Exception:
    summarize_pipe = None

def hf_sentiment(text):
    if sentiment_pipe is None:
        return {"error": "Sentiment model not available. Is transformers and torch installed?"}
    try:
        res = sentiment_pipe(text)
        return res[0]
    except Exception as e:
        return {"error": str(e)}

def hf_summarize(text):
    if summarize_pipe is None:
        return "Summarization model not available. Is transformers and torch installed?"
    try:
        summary = summarize_pipe(text, max_length=60, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"

def ollama_infer(prompt, model="phi"):
    try:
        url = "http://localhost:11434/api/generate"
        data = {"model": model, "prompt": prompt, "stream": False}
        r = requests.post(url, json=data)
        r.raise_for_status()
        result = r.json()
        return result.get("response", "No response")
    except Exception as e:
        return f"Ollama error: {e}"

def auto_ml(df, target):
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y_original = df[target]

    # Encode categorical predictors with get_dummies
    X = pd.get_dummies(X, drop_first=True)

    # Determine unique classes and type before transformation
    num_classes = y_original.nunique()
    y_is_float = pd.api.types.is_float_dtype(y_original)

    # Now encode y for classification if needed
    if y_original.dtype == 'object' or str(y_original.dtype).startswith('category'):
        y = pd.factorize(y_original)[0]
    else:
        y = y_original

    # Select model type
    if num_classes <= 20 and not y_is_float:
        model = RandomForestClassifier(random_state=42)
        task_type = "classification"
    else:
        y = pd.to_numeric(y, errors='raise')  # will raise if strings for regression
        model = RandomForestRegressor(random_state=42)
        task_type = "regression"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == "classification":
        score = accuracy_score(y_test, y_pred)
    else:
        mse = mean_squared_error(y_test, y_pred)
        score = sqrt(mse)  # RMSE (no 'squared' argument ever)

    return model, X_train, score, task_type

def generate_explanation(text):
    prompt = f"Explain this data analysis result: {text}"
    # Assuming Ollama server running locally
    return ollama_infer(prompt, model="phi")
