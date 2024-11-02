"""
Created on Sun Oct 20 12:13:18 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def swish(x, b):
    # Compute the exponential in a numerically stable way
    # np.clip is used to limit the values of b*x to avoid overflow
    z = np.clip(b * x, -500, 500)  # Clip to prevent overflow in exp
    return (x / (1 + np.exp(-z))).round(3)

def cost_function(U, forecasts, k, psi, N0, gamma):
    b = 50
    Nt = N0 + np.cumsum(U)-np.cumsum(forecasts, axis=1)
    return np.dot(swish(Nt,b)*k+swish(-Nt,b)*psi,gamma)

def expected_cost(U, forecasts, k, psi, N0, gamma):
    return np.mean(cost_function(
        U = U,
        forecasts = forecasts,
        k = k,
        psi = psi,
        N0 = N0,
        gamma = gamma
    ), axis = 0)

lambda_true = 3  # Mean (λ) of the Poisson distribution

time_Series_length = 300
forecast_horizon = 100
lead_time = 9
number_of_samples = 1000
ITE = 11

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

# Calculate the 10th and 90th percentiles for uncertainty
lower_bound = np.percentile(forecasts, 10, axis=0)
upper_bound = np.percentile(forecasts, 90, axis=0)

# Fill between the lower and upper bounds to indicate uncertainty
plt.fill_between(forecast_dates, lower_bound, upper_bound, color='green', alpha=0.2, label='Uncertainty Interval (10th-90th Percentile)')


plt.xlabel('Time', fontsize=14)  # X-axis label
plt.ylabel('Demand', fontsize=14)  # Y-axis label
plt.legend()
# Customize ticks and grid
plt.xticks(fontsize=12)  # X-axis ticks font size
plt.yticks(fontsize=12)  # Y-axis ticks font size
plt.grid(color='gray', linestyle='--', alpha=0.7)  # Add a grid for better readability




#%% Analytical expression

step = 1
possible_actions = np.arange(0,20)
t_forecast_horizon = np.arange(1,forecast_horizon+1)
k = 8*np.ones(len(t_forecast_horizon))
psi = 189*np.ones(len(t_forecast_horizon))
N0 = 37

gamma_0 = 0.95
gamma = gamma_0**t_forecast_horizon

storage_penalty = 1
percentile = psi[lead_time-1]/(psi[lead_time-1]+storage_penalty*k[lead_time-1])

print("percentile:", percentile)
print(max(np.percentile(forecasts[:,:lead_time].sum(axis = 1), 100*percentile)-N0,0))

decisions = np.zeros(forecast_horizon)
for t in range(lead_time, forecast_horizon+1):
    percentile = psi[t-1]/(psi[t-1]+k[t-1])
    decisions[t-1] = max(np.percentile(forecasts[:,:t].sum(axis = 1), 100*percentile)-N0,0)
decisions = new_arr = np.insert(np.diff(decisions).round().astype(int), 0, 0)
EC_greedy = expected_cost(decisions, forecasts, k, psi, N0, gamma)

print("Expected cost theoretical: ", EC_greedy)
#%%

def optimize(
        U_initial,
        forecasts,
        k,
        psi,
        N0,
        gamma,
        lead_time,
        possible_actions,
        step,
        optimization_steps
        ) -> tuple:
    costs = []
    U = U_initial.copy()
    EC_old = expected_cost(U, forecasts, k, psi, N0, gamma)
    costs.append(EC_old)
    for passes in range(optimization_steps):
        U_new = U.copy()
        for element in range(lead_time-1,len(U)):
            U_new[element] = possible_actions[np.clip(np.where(possible_actions == U[element])[0][0]+np.random.randint(-step, step+1),0,len(possible_actions) - 1)]
            EC_new = expected_cost(U_new, forecasts, k, psi, N0, gamma)
            if EC_new < EC_old:
                costs.append(EC_new)
                EC_old = EC_new
            else:
                U_new[element] = U[element]
        U = U_new
    return U, costs

U = np.zeros(forecast_horizon)
U_optimal, costs = optimize(decisions, forecasts, k, psi, N0, gamma, lead_time, possible_actions, step = step, optimization_steps= 50)

plt.figure()
plt.plot(costs)

print("minimum cost: ", min(costs))
print("actions: ", lead_time, U_optimal)
print("Current decision:", U_optimal[lead_time-1])

#%%

data = []
for step in tqdm (range(1,20)):
    data.append(optimize(
        np.zeros(forecast_horizon),
        forecasts,
        k,
        psi,
        N0,
        gamma,
        lead_time,
        possible_actions,
        step = 1,
        optimization_steps = 20
        ))


#%%

plt.figure()
plt.plot([i[1][-1] for i in data]+[EC_greedy])


plt.figure()
plt.plot([i[0][lead_time-1] for i in data]+[decisions[lead_time-1]])
