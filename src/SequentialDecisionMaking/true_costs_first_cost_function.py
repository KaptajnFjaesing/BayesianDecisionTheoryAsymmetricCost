"""
Created on Tue Oct 29 21:36:15 2024

@author: Jonas Petersen
"""

"""
Created on Tue Oct 29 20:42:38 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt

def swish(x, b):
    z = np.clip(b * x, -500, 500)
    return (x / (1 + np.exp(-z))).round(3)

def cost_function(U, forecasts, k_sc, k_uv, N0, gamma):
    b = 50
    Nt = N0 + np.cumsum(U)-np.cumsum(forecasts, axis=1)
    return np.dot(swish(Nt,b)*k_sc+swish(-Nt,b)*k_uv, gamma)

def expected_cost(U, forecasts, k_sc, k_uv, N0, gamma):
    return np.mean(cost_function(
        U = U,
        forecasts = forecasts,
        k_sc = k_sc, 
        k_uv = k_uv,
        N0 = N0,
        gamma = gamma
    ), axis = 0)

def actual_costs(N, k_sc, k_uv, b):
    return (swish(N,b)*k_sc+swish(-N,b)*k_uv).sum()

lambda_true = 3  # Mean (λ) of the Poisson distribution
b = 50
time_Series_length = 300
forecast_horizon = 52
lead_time = 6
number_of_samples = 1000
N0 = 37
k_sc = 5
k_uv = 20

actual_costs_baseline = []
actual_costs_by_hand = []
for iteration in range(1000):

    true_time_series = np.random.poisson(lambda_true, time_Series_length)
    training_dates = np.arange(len(true_time_series[:-forecast_horizon]))
    training_data = true_time_series[:-forecast_horizon]
    forecast_dates = len(true_time_series[:-forecast_horizon])+np.arange(len(true_time_series[-forecast_horizon:]))
    
    forecasts_model = np.random.poisson(np.mean(training_data), (number_of_samples,forecast_horizon))
    
    forecast_dates = forecast_dates[:forecast_horizon]
    forecasts = forecasts_model[:,:forecast_horizon]
    
    decisions_baseline = np.zeros(forecast_horizon)
    decisions_baseline[lead_time-1:] = round(np.mean(training_data))
    N_baseline = N0+np.cumsum(decisions_baseline)-np.cumsum(true_time_series[-forecast_horizon:])
    actual_costs_baseline.append(actual_costs(N_baseline, k_sc, k_uv,b))
    
    possible_decisions = np.arange(0,20)
    decisions_by_hand = np.zeros(forecast_horizon)
    criterion = k_uv/(k_uv+k_sc)
    for t in range(lead_time, forecast_horizon+1):
        N_t = forecasts[:,:t].sum(axis = 1) - decisions_by_hand[:t-1].sum()
        decisions_by_hand[t-1] = max(round(np.percentile(N_t, 100*criterion))-N0, 0)
    N_by_hand = N0+np.cumsum(decisions_by_hand)-np.cumsum(true_time_series[-forecast_horizon:])
    
    actual_costs_by_hand.append(actual_costs(N_by_hand, k_sc, k_uv, b))


    
plt.figure()
plt.hist(actual_costs_baseline, alpha = 0.5, bins = 50)
plt.hist(actual_costs_by_hand, alpha = 0.5, bins = 50)

print("mean baseline cost: ", np.mean(actual_costs_baseline))
print("mean by hand cost: ", np.mean(actual_costs_by_hand))

print("median baseline cost: ", np.median(actual_costs_baseline))
print("median by hand cost: ", np.median(actual_costs_by_hand))

"""
Note:
    - the baseline is almost perfect, since it is Natures model.

"""
