import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets.samples_generator import make_blobs

def pltPer(X, y, W):
    f = plt.figure()
    for n in range(len(y)):
        if y[n] == -1:
            plt.plot(X[n,1],X[n,2],'r*')
        else:
            plt.plot(X[n,1],X[n,2],'b.')
    m, b = -W[1]/W[2], -W[0]/W[2]
    l = np.linspace(min(X[:,1]),max(X[:,1])) #could just use max and min
    plt.plot(l, m*l+b, 'k-')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Perceptron Learning Algorithm")
    plt.show()

# Some attempt to do the PLA 
def main():
    N = 1000
    
    # data
    X, y = make_blobs(n_samples=N, centers=2, n_features=2) 
    y[y==0] = -1 #changing all the 0 to negative one 
    X = np.append(np.ones((N,1)), X, 1) #adding a column of ones
    

    # initialize the weigths to zeros
    w = np.zeros(3) #change this to 11 for 10 dims
    it = 0
    pltPer(X,y,w) # initial solution (bad!)  
    
    # Iterate until all points are correctly classified
    while classification_error(w, X, y) != 0:
        it += 1
        # Pick random misclassified point
        x, s = choose_miscl_point(w, X, y)
        # Update weights
        w += s*x
        
    pltPer(X,y,w) #commit this out before running 10 dims
    print("Total interations:" + str(it))

def classification_error(w, X, y):
    err_cnt = 0
    N = len(X)
    for n in range(N):
        s = np.sign(w.T.dot(X[n])) # if this is zero, then :(
        if y[n] != s:
            err_cnt += 1
    print(err_cnt)
    return err_cnt

def choose_miscl_point(w, X, y):
    mispts = []
    # Choose a random point among the misclassified
    for n in range(len(X)):
        if np.sign(w.T.dot(X[n])) != y[n]:
            mispts.append((X[n], y[n]))
    #print(len(mispts))
    return mispts[random.randrange(0,len(mispts))]

main()

