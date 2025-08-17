import pandas as pd
from core import visualization

def test_plot_histogram():
    df = pd.DataFrame({"numbers": [1, 2, 3, 4, 5]})
    # This should run without errors; no return value
    visualization.plot_histogram(df, "numbers")
