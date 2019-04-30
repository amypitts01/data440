import numpy as np
from matplotlib import pyplot as plt

def genDataSet(N):
    x = np.sort(np.random.normal(0, 1, N)*3)
    ytrue = (np.cos(x) + 2) / (np.cos(x * 1.4) + 2)
    noise = np.random.normal(0, 0.2, N)
    y = ytrue + noise
    X = np.c_[x, y]
    return X, ytrue
