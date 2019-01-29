import numpy as np 

#The two arrays given in question three
A = np.array([[1,4,-3],[2,-1,3]])
B = np.array([[-2,0,5],[0,-1,4]])

#Question 3.a
A.dot(B) #This shows that we can not multiply the two arrays
#Question 3.b
A.T.dot(B) #this is just the transpose
np.linalg.matrix_rank(A.T.dot(B))#this is the rank

