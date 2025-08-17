import pandas as pd
from core import stats_engine

def test_t_test():
    s1 = pd.Series([1, 2, 3, 4, 5])
    s2 = pd.Series([2, 3, 4, 5, 6])
    stats_engine.t_test(s1, s2)  # Should run without errors
