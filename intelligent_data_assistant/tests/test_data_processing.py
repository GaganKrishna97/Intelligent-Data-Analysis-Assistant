import pandas as pd
from core import data_processing

def test_detect_column_types():
    df = pd.DataFrame({
        "num": [1, 2],
        "cat": ["a", "b"],
        "dt": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "text": ["long text example", "another example"],
    })

    col_types = data_processing.detect_column_types(df)
    assert "num" in col_types["numeric"]
    assert "cat" in col_types["categorical"]
    assert "dt" in col_types["datetime"]
    assert "text" in col_types["text"]
