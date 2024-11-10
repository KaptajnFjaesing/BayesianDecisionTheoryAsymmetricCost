"""
Created on Thu Nov  7 14:42:39 2024

@author: Jonas Petersen
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


holding_costs = np.arange(0, 20+1, 1)
unit_values = np.arange(0, 100+1, 1)

with open("./src/SequentialDecisionMaking/costs_baseline.pkl", "rb") as f:
    costs_baseline = pickle.load(f)

with open("./src/SequentialDecisionMaking/costs_by_hand.pkl", "rb") as f:
    costs_by_hand = pickle.load(f)

ratio = costs_by_hand / costs_baseline

# %%
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Define the model function
def model_func(x, a, b):
    return np.sqrt(a + b * np.log(x))


params, covariance = curve_fit(model_func, unit_values[1:], ratio[:,1][1:], p0=[1, 1])
params2, covariance2 = curve_fit(model_func, 1/holding_costs[1:], ratio[1,:][1:], p0=[0.2, 0.01])

# Extract the fitted parameters
a_fit, b_fit = params
a_fit2, b_fit2 = params2
print(f"Fitted parameters: a = {a_fit}, b = {b_fit}")


#%%

c = 1
h = 1
ratio_analytical_c = np.sqrt(a_fit+b_fit*np.log(unit_values/h))
ratio_analytical_h = np.sqrt(a_fit2+b_fit2*np.log(c/holding_costs))


plt.figure()
plt.plot(ratio[:,1])
plt.plot(ratio_analytical_c)


plt.figure()
plt.plot(ratio[1,:])
plt.plot(ratio_analytical_h)
