import numpy as np
import matplotlib.pyplot as plt

def pltPer(X, y, W):
    for n in range(len(y)):
        if y[n] == -1:
            plt.plot(X[n,1],X[n,2],'ro')
        else:
            plt.plot(X[n,1],X[n,2],'bx')
    m, b = -W[1]/W[2], -W[0]/W[2]
    l = np.linspace(0,5)
    plt.plot(l, m*l+b, 'k-')

#Postive W
X = np.array([[1,1,1],
              [1,4,0],
              [1,2,-4],
              [1,3,-5]])
y = np.array([1, 1, -1, -1])
W = np.array([1,2,3]) #w0w1w2
pltPer(X,y,W)

plt.ylim(-6,2)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'The graph when $w=[1,2,3]^T$')
plt.savefig('hwplot_postive_w.pdf', bbox_inches='tight')
plt.show()

#Negative W
X = np.array([[1,1,1],
              [1,4,0],
              [1,2,-4],
              [1,3,-5]])
y = np.array([1, 1, -1, -1])
W = np.array([-1,-2,-3]) #w0w1w2
pltPer(X,y,W)

plt.ylim(-6,2)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'The graph when $w=-[1,2,3]^T$')
plt.savefig('hwplot_negative_w.pdf', bbox_inches='tight')
plt.show()