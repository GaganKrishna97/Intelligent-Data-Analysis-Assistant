from transformers import pipeline
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
import streamlit as st

# Hugging Face Transformers (local, free)
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
summarize_pipe = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def hf_sentiment(text):
    try:
        res = sentiment_pipe(text)
        return res[0]
    except Exception as e:
        return {"error": str(e)}

def hf_summarize(text):
    try:
        summary = summarize_pipe(text, max_length=60, min_length=10, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"

# Ollama lightweight models (local LLMs, no API key needed)
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

# AutoML functionality using sklearn
def auto_ml(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    if y.dtype == 'object' or len(y.unique()) < 20:
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if isinstance(model, RandomForestClassifier):
        report = classification_report(y_test, y_pred, output_dict=True)
        st.json(report)
    else:
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.4f}")

    return model, X_train

# Generate natural language explanation for results
def generate_explanation(text):
    prompt = f"Explain this data analysis result: {text}"
    response = ollama_infer(prompt, model="phi")  # Assumes ollama_infer exists
    return response
