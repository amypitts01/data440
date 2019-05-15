import finalGetDigits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import KFold

# get digits data X (training input) and y (target output)
X, y, X_te, y_te = finalGetDigits.getDataSet()

#penC  <- Penalty parameter C of the error term
#tubEpsilon  <- the epsilon-tube within which no penalty is associated

bestC=0
bestEpsilon=0
bestGamma=0
bestScore=float('-inf')
score=0
for penC in np.logspace(6, 12, num=7, base=2):
  for tubEpsilon in np.linspace(0.5, 2.5, num=21):
    for paramGamma in np.logspace(-6, -2, num=5, base=2):
      kf = KFold(n_splits=np.random.randint(2,11))
      cvscore=[]
      for train, validation in kf.split(X):
        X_train, X_validation, y_train, y_validation = X[train, :], X[validation, :], y[train], y[validation]
        # here we create the SVR
        svr =  SVR(C=penC, epsilon=tubEpsilon, gamma=paramGamma, kernel='rbf', verbose=False)
        # here we train the SVR
        svr.fit(X_train, y_train)
        # now we get E_out for validation set
        score=svr.score(X_validation, y_validation)
        cvscore.append(score)

      # average CV score
      score=sum(cvscore)/len(cvscore)
      if (score > bestScore):
        bestScore=score
        bestC=penC
        bestEpsilon=tubEpsilon
        bestGamma=paramGamma
        print("BEST! -> C " + str(penC) + ", epsilon " + str(tubEpsilon) + ", gamma " + str(paramGamma) + ". Testing set CV score: %f" % score)
      else:
        print("C " + str(penC) + ", epsilon " + str(tubEpsilon) + ", gamma " + str(paramGamma) + ". Testing set CV score: %f" % score)

# here we create the final SVR
svr =  SVR(C=bestC, epsilon=bestEpsilon, gamma=bestGamma, kernel='rbf', verbose=True)
# here we train the final SVR
svr.fit(X, y)
# E_out in training
print("Training set score: %f" % svr.score(X, y)) 
# here test the final SVR and get E_out for testing set
ypred=svr.predict(X_te)
score=svr.score(X_te, y_te)
print("Testing set score: %f" % score)

x_min, x_max = np.min(X_te, axis=0), np.max(X_te, axis=0)
X_te = (X_te - x_min) / (x_max - x_min)

plt.figure(figsize=(6, 4))
for i in range(X_te.shape[0]):
  plt.text(X_te[i, 0], X_te[i, 1], str(y_te[i]), color=plt.cm.spectral(round(ypred[i]) / 10.), fontdict={'weight': 'bold', 'size': 9})

plt.xticks([])
plt.yticks([])
plt.axis('off')
plt.tight_layout()

plt.show()
