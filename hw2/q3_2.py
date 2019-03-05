import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets.samples_generator import make_blobs

#Data 
def make_semi_circles(n_samples=2000, thk=5, rad=10, sep=5, plot=True):
    noisey = np.random.uniform(low=-thk/100.0, high=thk/100.0, size=(n_samples // 2))
    noisex = np.random.uniform(low=-rad/100.0, high=rad/100.0, size=(n_samples // 2))
    separation = np.ones(n_samples // 2)*((-sep*0.1)-0.6)
    
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
 
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out)) + noisex
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out)) + noisey
    inner_circ_x = (1 - np.cos(np.linspace(0, np.pi, n_samples_in))) + noisex
    inner_circ_y = (1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5) + noisey + separation
    
    X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.ones(n_samples_in, dtype=np.intp)*-1,
                   np.ones(n_samples_out, dtype=np.intp)])
    return X, y


# PLA 
l = list(np.arange(0.2,5,0.2)) #creating the numbers 
itlist = [] #empty array to saving the iterations 

for i in range(len(l)):
    def main():
        N=2000
        # data
        X,y = make_semi_circles(n_samples=2000, thk=5, rad=10, sep=l[i], plot=True)
        X = np.append(np.ones((N,1)), X, 1) #adding a column of ones
        
        # initialize the weigths to zeros
        w = np.zeros(3) #change this to 11 for 10 dims
        it = 0
    
        # Iterate until all points are correctly classified
        while classification_error(w, X, y) != 0:
            it += 1
            x, s = choose_miscl_point(w, X, y) # Pick random misclassified point
            w += s*x # Update weights
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

plt.plot(l,itlist, 'ro') #plotting the iterations against sep length
plt.title("Number of Iterations Changing sep length")
plt.xlabel("sep length")
plt.ylabel("Number of Iterations")
plt.show()



