import numpy as np
import matplotlib.pyplot as plt
import random
#from sklearn.datasets.samples_generator import make_blobs
from numpy import genfromtxt

#plotting
def pltPer(X, y, W):
    # plots data 
    c0 = plt.scatter(X[y==-1,1], X[y==-1,2], s=20, color='r', marker='x')
    c1 = plt.scatter(X[y==1,1], X[y==1,2], s=20, color='b', marker='o')
    # displays legend 
    plt.legend((c0, c1), ('All Other Numbers -1', 'Number Zero +1'), 
            loc='upper right', scatterpoints=1, fontsize=11)
    # displays axis legends and title
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.title(r'Intensity and Symmetry of Digits')
    m, b = -W[1]/W[2], -W[0]/W[2]
    
    l = np.linspace(min(X[:,1]),max(X[:,1])) #could just use max and min
    plt.plot(l, m*l+b, 'k-')
    #plt.savefig('midterm_plot.pdf', bbox_inches='tight')
    plt.show()

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

# PLA 

def main():
    # data  
    # read digits data & split it into X (training input) and y (target output)
    dataset = genfromtxt('features.csv', delimiter=' ')
    y = dataset[:, 0]
    X = dataset[:, 1:]
    y[y!=4] = -1    #rest of numbers are negative class
    y[y==4] = +1    #number zero is the positive class
    N = len(y)
    X = np.append(np.ones((len(y),1)), X, 1)

    #linear regression
    Xs = np.linalg.pinv(X.T.dot(X)).dot(X.T) 
    wlr = Xs.dot(y)
    
    #pltPer(X,y,wlr) 
    
    # initialize the weigths to zeros
    w = np.zeros(3) #this is for the pocket algorithm starting from w=0
    it = 0
    pltPer(X,y,w)  # initial solution (bad!)
    #w = wlr
    
    stopIt = 25
    currIt = 0
    bestW = w
    bestE = N
    # Iterate until all points are correctly classified
    nerr = classification_error(w, X, y)
    while nerr != 0:
        it += 1
        currIt += 1
        if currIt > stopIt:
            print("early stop")
            break
        # Pick random misclassified point
        x, s = choose_miscl_point(w, X, y)
        # Update weights
        w += s*x
        nerr = classification_error(w, X, y)
        if nerr < bestE:
            currIt = 0
            bestE = nerr
            bestW = w
        
    w = bestW
    pltPer(X,y,w)
    print("Total iterations: " + str(it))
    print("Classification error: " + str(nerr))

main()
