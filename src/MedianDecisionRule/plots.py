# -*- coding: utf-8 -*-
#%%
"""
Created on Thu Jun 13 18:21:35 2024

@author: roman
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x, beta):
    return x * sigmoid(beta * x)

def custom_function(x, alpha, beta):
    term1 = alpha * swish(x, beta)
    term2 = (1 - alpha) * swish(-x, beta)
    return term1 + term2

# Parameters
beta = 50
X = np.linspace(-5, 5, 400)  # Range for U_i(D)


# Plotting the function
plt.figure(figsize=(10, 6))
plt.plot(X, [custom_function(x, 0.1, beta) for x in X], label=r'$(\alpha\cdot \text{swish}(z,\beta)+(1-\alpha)\cdot\text{swish}(-z,\beta))|_{\beta=50,\alpha = 0.1}$')
plt.plot(X, [custom_function(x, 0.2, beta) for x in X], label=r'$(\alpha\cdot \text{swish}(z,\beta)+(1-\alpha)\cdot\text{swish}(-z,\beta))|_{\beta=50,\alpha = 0.2}$')
plt.plot(X, [custom_function(x, 0.7, beta) for x in X], label=r'$(\alpha\cdot \text{swish}(z,\beta)+(1-\alpha)\cdot\text{swish}(-z,\beta))|_{\beta=50,\alpha = 0.7}$')
plt.xlabel(r'$z$')
plt.ylabel('Cost')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('../docs/figures/cost_plot.pdf')

# %%

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters for the normal distribution
mu = 0  # mean
sigma = 1  # standard deviation

# Define the probability density function
def p(s_i, mu, sigma):
    return stats.norm.pdf(s_i, mu, sigma)

# Define the integral boundary
U = 1.64  # Example value for U_i(D)

# Create a range of values for s_i
s_i_range = np.linspace(-4, 4, 1000)

# Calculate the probability density function values
p_values = p(s_i_range, mu, sigma)

# Calculate the alpha value (area under the curve from U_i(D) to infinity)
alpha = stats.norm.sf(U, mu, sigma)  # survival function for the upper tail
print("alpha=",alpha)

# Plotting the density function
plt.figure(figsize=(10, 6))
plt.plot(s_i_range, p_values, label=r'$p(s|x,D,I)$')
plt.fill_between(s_i_range, p_values, where=(s_i_range > U_i_D), color='gray', alpha=0.5, label=r'$\alpha$ area')
plt.axvline(U_i_D, color='red', linestyle='--', label=r'$U^*$')

# Annotate the plot
plt.xlabel(r'$s$')
plt.ylabel(r'$p(s|x,D,I)$')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig('../docs/figures/alpha_plot.pdf')
