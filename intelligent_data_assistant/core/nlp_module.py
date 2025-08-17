import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import pandas as pd

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except:
        nltk.download("stopwords")

ensure_nltk()

def get_keywords(text_series):
    alltext = " ".join(text_series.dropna().astype(str).tolist()).lower()
    tokens = [t for t in word_tokenize(alltext) if t.isalpha()]
    stops = set(stopwords.words("english"))
    freq = nltk.FreqDist(t for t in tokens if t not in stops)
    return freq.most_common(10)

def get_sentiment(text):
    try:
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        tb = TextBlob(text)
        return tb.sentiment.polarity
    except:
        return 0.0
