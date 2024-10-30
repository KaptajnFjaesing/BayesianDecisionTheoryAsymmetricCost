"""
Created on Tue Oct 29 20:42:38 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing

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

lambda_true = 3  # Mean (λ) of the Poisson distribution

time_Series_length = 300
forecast_horizon = 52
lead_time = 9
number_of_samples = 1000
ITE = 20

np.random.seed(ITE)
true_time_series = np.random.poisson(lambda_true, time_Series_length)
training_dates = np.arange(len(true_time_series[:-forecast_horizon]))
training_data = true_time_series[:-forecast_horizon]
forecast_dates = len(true_time_series[:-forecast_horizon])+np.arange(len(true_time_series[-forecast_horizon:]))

np.random.seed(ITE)

forecasts_model = np.random.poisson(np.mean(training_data), (number_of_samples,forecast_horizon))

forecast_dates = forecast_dates[:forecast_horizon]
forecasts = forecasts_model[:,:forecast_horizon]

median_forecasts = np.median(forecasts, axis = 0)

plt.figure(figsize = (20,10))
plt.plot(training_dates,training_data, label = "Training Data")
plt.plot(forecast_dates,true_time_series[-forecast_horizon:], label = "Test Data")
plt.plot(forecast_dates,median_forecasts, label = "Forecast")

# Calculate the 0.1 and 0.9 quantile for uncertainty
lower_bound = np.quantile(forecasts, 0.1, axis=0)
upper_bound = np.quantile(forecasts, 0.9, axis=0)

# Fill between the lower and upper bounds to indicate uncertainty
plt.fill_between(forecast_dates, lower_bound, upper_bound, color='green', alpha=0.2, label='Uncertainty Interval (10th-90th Percentile)')


plt.xlabel('Time', fontsize=14)  # X-axis label
plt.ylabel('Demand', fontsize=14)  # Y-axis label
plt.legend()
# Customize ticks and grid
plt.xticks(fontsize=12)  # X-axis ticks font size
plt.yticks(fontsize=12)  # Y-axis ticks font size
plt.grid(color='gray', linestyle='--', alpha=0.7)  # Add a grid for better readability


#%% Different approaches
from scipy.optimize import root_scalar
import pandas as pd
import time

k_sc = 5
k_uv = 20
step = 1
possible_actions = np.arange(0, 20)
t_forecast_horizon = np.arange(1, forecast_horizon+1)
N0 = 37

gamma_0 = 1
gamma = gamma_0**t_forecast_horizon
tic = time.time()

decisions_baseline = np.zeros(forecast_horizon)
decisions_baseline[lead_time-1:] = round(np.mean(training_data))
toc = time.time()
EC_baseline = expected_cost(decisions_baseline, forecasts, k_sc, k_uv, N0, gamma)

print("Expected cost baseline: ", EC_baseline, "; computation time", round(toc-tic,4))

tic = time.time()
possible_decisions = np.arange(0,20)
decisions_by_hand = np.zeros(forecast_horizon)
criterion = k_uv/(k_uv+k_sc)
for t in range(lead_time, forecast_horizon+1):
    N_t = forecasts[:,:t].sum(axis = 1) - decisions_by_hand[:t-1].sum()
    decisions_by_hand[t-1] = max(round(np.quantile(N_t, criterion))-N0, 0)
toc = time.time()
EX_by_hand = expected_cost(decisions_by_hand, forecasts, k_sc, k_uv, N0, gamma)
print("Expected cost by hand: ", EX_by_hand, "; computation time", round(toc-tic,4))

def derivative(U, forecasts, t, N_old, k_sc, k_uv):
    N_t = N_old+U - forecasts[:,:t].sum(axis = 1)
    condition = (N_t >= 0).astype(int)
    return ((k_sc + k_uv) * condition - k_uv).mean()

tic = time.time()
decisions_root_scalar = np.zeros(forecast_horizon)
for t in range(lead_time,forecast_horizon+1):
    try:
        decisions_root_scalar[t-1] = root_scalar(derivative, args=(forecasts, t, N0+decisions_root_scalar[:t-1].sum(), k_sc, k_uv), bracket= [0, 20]).root
    except:
        decisions_root_scalar[t-1] = 0

toc = time.time()
EC_root_scalar = expected_cost(decisions_root_scalar, forecasts, k_sc, k_uv, N0, gamma)
print("Expected cost root_scalar: ", EC_root_scalar, "; computation time", round(toc-tic,4))

def optimize(
        U_initial,
        forecasts,
        k_sc,
        k_uv,
        N0,
        gamma,
        lead_time,
        possible_actions,
        step,
        optimization_steps
        ) -> tuple:
    costs = []
    U = U_initial.copy()
    EC_old = expected_cost(U, forecasts, k_sc, k_uv, N0, gamma)
    costs.append(EC_old)
    for passes in range(optimization_steps):
        U_new = U.copy()
        for element in range(lead_time-1,len(U)):
            U_new[element] = possible_actions[np.clip(np.where(possible_actions == U[element])[0][0]+np.random.randint(-step, step+1),0,len(possible_actions) - 1)]
            EC_new = expected_cost(U_new, forecasts, k_sc, k_uv, N0, gamma)
            if EC_new < EC_old:
                costs.append(EC_new)
                EC_old = EC_new
            else:
                U_new[element] = U[element]
        U = U_new
    return U, costs

tic = time.time()
decisions_RW, costs = optimize(np.zeros(forecast_horizon), forecasts, k_sc, k_uv, N0, gamma, lead_time, possible_actions, step = step, optimization_steps = 100)
toc = time.time()
EC_RW = expected_cost(decisions_RW, forecasts, k_sc, k_uv, N0, gamma)
plt.figure()
plt.plot(costs)
print("Expected cost RW: ", EC_RW, "; computation time", round(toc-tic,4))

tic = time.time()
bounds = [(0, 20) if x > lead_time - 2 else (0, 1e-9) for x in range(forecast_horizon)]

additional_args = (forecasts, k_sc, k_uv, N0, gamma)
result = dual_annealing(
    expected_cost,
    bounds = bounds,
    args = additional_args,
    maxiter = 1000,
    maxfun = 1000
)
decisions_dual_annealing = result.x.round().astype(int)
toc = time.time()
EC_dual_annealing = expected_cost(decisions_dual_annealing, forecasts, k_sc, k_uv, N0, gamma)
print("Expected cost dual_annealing:", EC_dual_annealing,  "; computation time", round(toc-tic,4))


df_decisions = pd.DataFrame(data = list(zip(decisions_baseline,decisions_by_hand,decisions_root_scalar,decisions_RW,decisions_dual_annealing)), columns = ["Baseline","By hand","root_scalar", "Numerical", "dual_annealing"])


# Actual costs

def actual_costs(N, k_sc, k_uv):
    return (swish(N,b)*k_sc+swish(-N,b)*k_uv).sum()

b = 50
N_baseline = N0+np.cumsum(decisions_baseline)-np.cumsum(true_time_series[-forecast_horizon:])
N_by_hand = N0+np.cumsum(decisions_by_hand)-np.cumsum(true_time_series[-forecast_horizon:])
N_root_scalar = N0+np.cumsum(decisions_root_scalar)-np.cumsum(true_time_series[-forecast_horizon:])
N_rw = N0+np.cumsum(decisions_RW)-np.cumsum(true_time_series[-forecast_horizon:])
N_dual_annealing = N0+np.cumsum(decisions_dual_annealing)-np.cumsum(true_time_series[-forecast_horizon:])

print("Baseline costs: ", actual_costs(N_baseline, k_sc, k_uv))
print("By hand costs: ", actual_costs(N_by_hand, k_sc, k_uv))
print("Root scalar costs: ", actual_costs(N_root_scalar, k_sc, k_uv))
print("RW costs: ", actual_costs(N_rw, k_sc, k_uv))
print("Dual annealing costs: ", actual_costs(N_dual_annealing, k_sc, k_uv))
