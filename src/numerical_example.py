"""
Created on Thu Oct 10 18:37:42 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import dual_annealing

def swish(x, b):
    z = np.clip(b * x, -500, 500)
    return (x / (1 + np.exp(-z)))

def cost_function(s, U, a):
    b = 200
    return a*swish(U-s,b)+(1-a)*swish(s-U,b)

def expected_cost(forecasts, u, a):
    return np.mean(cost_function(
        s = forecasts,
        U = u,
        a = a
    ))

gamma_sample = np.random.gamma(2, 1, size = 100000)

bounds = [(0, 10)]

numerical_results = []
analytical_results = []
for a in tqdm(np.arange(0.05, 1, 0.05)): 
    additional_args = (gamma_sample, a)
    numerical_result = dual_annealing(
        expected_cost,
        bounds = bounds,
        args = additional_args
    )
    numerical_results.append(numerical_result.x[0])
    analytical_results.append(np.percentile(gamma_sample, 100*a, axis=0))

#%%
plt.figure(figsize=(12, 8), constrained_layout=True)
plt.plot(np.arange(0.05,1,0.05),(np.array(numerical_results)-np.array(analytical_results))/np.array(analytical_results))
plt.grid(True)
plt.xlabel('a', fontsize=12)
plt.xlim([0,1])
plt.ylabel('Relative difference between numerical and analytical predictions', fontsize=12)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
plt.savefig('./docs/figures/numerical_example.pdf')

