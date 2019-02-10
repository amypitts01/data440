import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets.samples_generator import make_blobs

# Some attempt to do the PLA 
itlist = []
for x in range(100):
    def main():
        N = 1000
        
        # data
        X, y = make_blobs(n_samples=N, centers=2, n_features=10) 
        y[y==0] = -1 #changing all the 0 to negative one 
        X = np.append(np.ones((N,1)), X, 1) #adding a column of ones
        
        # initialize the weigths to zeros
        w = np.zeros(11) 
        it = 0
        
        # Iterate until all points are correctly classified
        while classification_error(w, X, y) != 0:
            it += 1
            # Pick random misclassified point
            x, s = choose_miscl_point(w, X, y)
            # Update weights
            w += s*x
        itlist.append(it) #saving the iterations

    def classification_error(w, X, y):
        err_cnt = 0
        N = len(X)
        for n in range(N):
            s = np.sign(w.T.dot(X[n])) # if this is zero, then :(
            if y[n] != s:
                err_cnt += 1
        return err_cnt

    def choose_miscl_point(w, X, y):
        mispts = []
        # Choose a random point among the misclassified
        for n in range(len(X)):
            if np.sign(w.T.dot(X[n])) != y[n]:
                mispts.append((X[n], y[n]))
        return mispts[random.randrange(0,len(mispts))]

    main()

plt.hist(itlist) #plotting a histogram 
plt.title("Number of Iterations")
plt.xlabel("Number")
plt.ylabel("Frequency")
plt.show()


