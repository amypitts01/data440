import numpy as np
import matplotlib.pyplot as plt

all_n = [] #empty list to save values of N
n = 20000 #initial starting place 
all_n.append(n) 
Sn = 3200 * np.log(81920*n**float(10)+80) #second N
all_n.append(Sn)

t=True
while(t):
    if(round(Sn,8) == round(n,8)): #checking if equal
        t=False
        print(Sn) #if equal then print 
    else: #if not equal then recalculate 
        n = Sn
        Sn = 3200 * np.log(81920*n**float(10)+80)
        all_n.append(Sn)
#plot the Ns 
iterations = list(range(len(all_n))) #creating an x-axis
plt.plot(iterations,all_n, 'ro') #plotting the n against iteration 
plt.title("N against Iterations")
plt.xlabel("Iteration")
plt.ylabel("N")
plt.show()
