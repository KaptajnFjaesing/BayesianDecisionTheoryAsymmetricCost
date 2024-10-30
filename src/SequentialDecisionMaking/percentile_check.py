"""
Created on Wed Oct 30 08:07:54 2024

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

true_time_series = np.random.poisson(lambda_true, time_Series_length)
training_dates = np.arange(len(true_time_series[:-forecast_horizon]))
training_data = true_time_series[:-forecast_horizon]
forecast_dates = len(true_time_series[:-forecast_horizon])+np.arange(len(true_time_series[-forecast_horizon:]))

forecasts_model = np.random.poisson(np.mean(training_data), (number_of_samples,forecast_horizon))

forecast_dates = forecast_dates[:forecast_horizon]
forecasts = forecasts_model[:,:forecast_horizon]

possible_decisions = np.arange(0,20)
decisions_by_hand = np.zeros(forecast_horizon)
decisions_by_hand2 = np.zeros(forecast_horizon)
criterion = k_uv/(k_uv+k_sc)
for t in range(lead_time, forecast_horizon+1):
    N_t = forecasts[:,:t].sum(axis = 1) - decisions_by_hand[:t-1].sum()
    decisions_by_hand[t-1] = max(round(np.percentile(N_t-N0, 100*criterion)), 0)

    N_t2 = N0 + possible_decisions[:,None] - forecasts[:,:t].sum(axis = 1)+decisions_by_hand[:t-1].sum()
    condition = (N_t2 >= 0).astype(int)
    gradient = ((k_sc + k_uv) * condition - k_uv).mean(axis = 1)
    minimum = np.where(gradient == 0)[0]
    if len(minimum) > 0:
        decisions_by_hand2[t-1] = possible_decisions[minimum[0]]
    else:
        sign_change_idx = np.where((gradient[:-1] < 0) & (gradient[1:] > 0))[0]
        if len(sign_change_idx) > 0:
            decisions_by_hand2[t-1] = possible_decisions[sign_change_idx[0]+1]
        else:
            decisions_by_hand2[t-1] = possible_decisions[np.argmin(gradient)]

gamma_0 = 1
t_forecast_horizon = np.arange(1, forecast_horizon+1)
gamma = gamma_0**t_forecast_horizon
EX_by_hand = expected_cost(decisions_by_hand, forecasts, k_sc, k_uv, N0, gamma)
EX_by_hand2 = expected_cost(decisions_by_hand2, forecasts, k_sc, k_uv, N0, gamma)
  
print(EX_by_hand)
print(EX_by_hand2)

#%%

decisions_by_hand = np.zeros(forecast_horizon)
decisions_by_hand2 = np.zeros(forecast_horizon)
criterion = k_uv/(k_uv+k_sc)
#%%
t = 13
N_t = forecasts[:,:t].sum(axis = 1) - decisions_by_hand[:t-1].sum()
decisions_by_hand[t-1] = max(round(np.percentile(N_t, 100*criterion))-N0, 0)

N_t2 = N0 + possible_decisions[:,None] - forecasts[:,:t].sum(axis = 1)+decisions_by_hand[:t-1].sum()
condition = (N_t2 >= 0).astype(int)
gradient = ((k_sc + k_uv) * condition - k_uv).mean(axis = 1)

minimum = np.where(gradient == 0)[0]
if len(minimum) > 0:
    decisions_by_hand2[t-1] = possible_decisions[minimum]
    
sign_change_idx = np.where((gradient[:-1] < 0) & (gradient[1:] > 0))[0]
if len(sign_change_idx) > 0:
    decisions_by_hand2[t-1] = possible_decisions[sign_change_idx[0]+1]
else:
    decisions_by_hand2[t-1] = possible_decisions[np.argmin(gradient)]

print(decisions_by_hand)
print(decisions_by_hand2)