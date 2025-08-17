import pandas as pd
from core import nlp_module

def test_analyze_sentiment():
    df = pd.DataFrame({"text": ["I love this", "I hate that"]})
    # This should run without errors; no return value
    nlp_module.analyze_sentiment(df, "text")
