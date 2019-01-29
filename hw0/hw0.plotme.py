# file: driver.py
# author: Dr. Pablo Rivas
# version: 1.0
# date: Aug/5/2016
# 
# This file shows how to create a simple plot in python 
# using the well-known matplotlib package.

import numpy as np
import matplotlib.pyplot as plt

# Creates an evenly sampled time array at 100ms intervals
t = np.arange(0., 5., 0.1)

plt.plot(t, t, 'r--')     # Uses red dashes for t
plt.plot(t, t**2, 'bs')   # Uses blue squares for t^2
plt.plot(t, t**3, 'g^')   # Uses green triangles for t^3
plt.plot(t, t**3.3, 'k-') # Uses black line for t^3.3

# Example axis legends title
plt.xlabel(r'This is time $t$')
plt.ylabel(r'The units of $f(t)$')
plt.title(r'The title accepts math: $f(t)=\sigma_i=15$')

# Saves the figure into a .pdf file (desired!)
plt.savefig('hwplot.pdf', bbox_inches='tight')
plt.show()
