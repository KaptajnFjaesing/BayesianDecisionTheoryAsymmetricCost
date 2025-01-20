"""
Created on Wed Nov  6 18:57:26 2024

@author: Jonas Petersen
"""
#%%
import numpy as np
from scipy.stats import poisson
from scipy.special import gammaincc, gamma

def test_poisson_cdf_vs_gamma_ratio(k, lambda_):
    # Compute the Poisson CDF at k for parameter lambda_
    poisson_cdf = poisson.cdf(k, lambda_)

    # Compute the ratio of the incomplete gamma function to the complete gamma function
    gamma_ratio = gammaincc(k + 1, lambda_)  # This is Γ(k+1, λ) / Γ(k+1)

    # Print results
    print(f"Poisson CDF (P(X <= {k}) with λ={lambda_}): {poisson_cdf}")
    print(f"Gamma Ratio (γ({k+1}, {lambda_}) / Γ({k+1})): {gamma_ratio}")
    print(f"Difference: {abs(poisson_cdf - gamma_ratio)}")

    # Test if they are approximately equal
    return np.isclose(poisson_cdf, gamma_ratio, atol=1e-10)

# Test the function with different values of k and lambda
k_values = [0, 1, 5, 10]
lambda_values = [0.5, 1.0, 5.0, 10.0]

for k in k_values:
    for lambda_ in lambda_values:
        result = test_poisson_cdf_vs_gamma_ratio(k, lambda_)
        print(f"Are Poisson CDF and Gamma Ratio approximately equal for k={k}, λ={lambda_}? {'Yes' if result else 'No'}\n")

#%%

import numpy as np
from scipy.stats import poisson

def truncated_poisson_mean(z, lambda_):
    """Compute the truncated mean sum_{k=0}^{z} k * p(k|D, I) for a Poisson distribution."""
    k_values = np.arange(0, z + 1)
    probabilities = poisson.pmf(k_values, lambda_)
    truncated_mean = np.sum(k_values * probabilities)
    return truncated_mean

# Example values for z and lambda
z_values = [5, 10, 15]
lambda_values = [1.0, 3.0, 5.0]

for z in z_values:
    for lambda_ in lambda_values:
        result = truncated_poisson_mean(z, lambda_)
        print(result-lambda_*gammaincc(z, lambda_))
        
        
#%%


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammainc

def gamma_expression(x, y):
    # Calculate (x - y) * Gamma(x, y) + y^x * e^(-y)
    upper_incomplete_gamma = gamma(x) * (1 - gammainc(x, y))  # Gamma(x, y)
    gamma_term = (x - y) * upper_incomplete_gamma/gamma(x)
    exp_term = (y ** x) * np.exp(-y)/gamma(x)
    return gamma_term, exp_term, gamma_term+exp_term

# Define ranges for x and y
x_values = np.linspace(1, 10, 10)
y_values = np.linspace(1, 10, 10)

# Create a meshgrid for plotting
X, Y = np.meshgrid(x_values, y_values)
Z = gamma_expression(X, Y)



#%%
# Plot the result
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Value of (x - y) * Gamma(x, y) + y^x * e^(-y)')
plt.xlabel("x")
plt.ylabel("y")
plt.title(r"Plot of $(x - y) \Gamma(x, y) + y^x e^{-y}$")
plt.show()


# %%
