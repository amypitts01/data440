# By Dr. Rivas (R) 2016

import matplotlib.pyplot as plt
import numpy as np

def make_semi_circles(n_samples=2000, thk=5, rad=10, sep=5, plot=True):
    #"""Make two semicircles circles
    #A simple toy dataset to visualize classification algorithms.
    #Parameters
    #----------
    #n_samples : int, optional (default=2000)
    #    The total number of points generated.
    #thk : int, optional (default=5)
    #    Thickness of the semi circles.
    #rad : int, optional (default=10)
    #    Radious of the circle.
    #sep : int, optional (default=5)
    #    Separation between circles.
    #plot : boolean, optional (default=True)
    #    Whether to plot the data.
    #Returns
    #-------
    #X : array of shape [n_samples, 2]
    #    The generated samples.
    #y : array of shape [n_samples]
    #    The integer labels (-1 or 1) for class membership of each sample.
    #"""
    
    noisey = np.random.uniform(low=-thk/100.0, high=thk/100.0, size=(n_samples // 2))
    
    noisex = np.random.uniform(low=-rad/100.0, high=rad/100.0, size=(n_samples // 2))
    
    separation = np.ones(n_samples // 2)*((-sep*0.1)-0.6)
    

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    # generator = check_random_state(random_state)
    
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out)) + noisex
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out)) + noisey
    inner_circ_x = (1 - np.cos(np.linspace(0, np.pi, n_samples_in))) + noisex
    inner_circ_y = (1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - .5) + noisey + separation
    
    X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.ones(n_samples_in, dtype=np.intp)*-1,
                   np.ones(n_samples_out, dtype=np.intp)])
    
    if plot:
        plt.plot(outer_circ_x, outer_circ_y, 'r.')
        plt.plot(inner_circ_x, inner_circ_y, 'b.')
        plt.show()

      
    return X, y

X, y = make_semi_circles(n_samples=2000, thk=5, rad=10, sep=5, plot=True)

print(X)
print(y)