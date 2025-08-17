import numpy as np
from scipy import stats

def t_test_ind(a, b):
    tval, pval = stats.ttest_ind(a.dropna(), b.dropna(), equal_var=False)
    return {"t": tval, "p": pval}
