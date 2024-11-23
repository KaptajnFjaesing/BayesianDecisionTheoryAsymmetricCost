"""
Created on Tue Nov 19 19:10:31 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaincc
from scipy.stats import poisson

def gamma_ratio(x,y):
    return gammaincc(x + 1, y)

def poisson_sum(x, y):
    return sum([poisson.pmf(k, y) for k in range(x + 1)])

def P1(x,y):
    m = 1.8
    e1 = -m*(x-y+1)/np.sqrt(y)
    return 1/(1+np.exp(e1))

def plot_function(f, cmap='viridis'):
    # Generate a grid of x and y values
    x = np.arange(0,100+1,1).astype(int)
    y = np.arange(0,100+1,1)
    X, Y = np.meshgrid(x, y)  # Create a 2D grid
    
    Z = np.array([[f(x1,y1) for x1 in x] for y1 in y])
    
    # Create the color plot
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(X, Y, Z, shading='auto', cmap=cmap)
    plt.colorbar(label='f(x, y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Color Plot of f(x, y)')
    plt.tight_layout()
    plt.show()

plot_function(f = gamma_ratio)
plot_function(f = P1)
plot_function(f = poisson_sum)
