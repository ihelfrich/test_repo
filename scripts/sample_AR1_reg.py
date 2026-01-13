# Create an AR1 model and sample from it
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def generate_ar1_data(phi: float, sigma: float, n: int, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    epsilons = np.random.normal(0, sigma, n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + epsilons[t]
    return pd.DataFrame({"time": np.arange(n), "value": y})