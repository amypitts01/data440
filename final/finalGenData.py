import numpy as np

def genDataSet(N):
    x = np.sort(np.random.normal(0, 1, N)*6)
    ytrue = np.sinc(x)
    noise = np.random.normal(0, 0.2, N)
    y = ytrue + noise
    X = np.c_[x, y]
    return X, ytrue
