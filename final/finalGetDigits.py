import numpy as np
from numpy import genfromtxt

def getDataSet():
  # read digits data & split it into X and y for training and testing
  dataset = genfromtxt('features.csv', delimiter=' ')
  y = dataset[:, 0]
  X = dataset[:, 1:]

  dataset = genfromtxt('features-t.csv', delimiter=' ')
  y_te = dataset[:, 0]
  X_te = dataset[:, 1:]
  return X, y, X_te, y_te
