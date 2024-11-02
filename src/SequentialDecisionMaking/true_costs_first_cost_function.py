"""
Created on Tue Oct 29 21:36:15 2024

@author: Jonas Petersen
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def relu(x):
    return np.maximum(x,0)

def cost_function(U, forecasts, h, c, N0, gamma):
    Nt = N0 + np.cumsum(U)-np.cumsum(forecasts, axis=1)
    return np.dot(relu(Nt)*h+relu(-Nt)*c, gamma)

def expected_cost(U, forecasts, h, c, N0, gamma):
    return np.mean(cost_function(
        U = U,
        forecasts = forecasts,
        h = h, 
        c = c,
        N0 = N0,
        gamma = gamma
    ), axis = 0)

def actual_costs(N, h, c):
    return (relu(N)*h+relu(-N)*c).sum()

def generate_data(lambda_true, time_Series_length, forecast_horizon, number_of_samples):
    true_time_series = np.random.poisson(lambda_true, time_Series_length)
    training_data = true_time_series[:-forecast_horizon]
    forecasts_model = np.random.poisson(np.mean(training_data), (number_of_samples,forecast_horizon))
    return training_data, forecasts_model[:,:forecast_horizon], true_time_series

lambda_true = 3  # Mean (λ) of the Poisson distribution
possible_decisions = np.arange(0,20)
time_Series_length = 300
forecast_horizon = 52
lead_time = 6
number_of_samples = 2000
N0 = 37
h = 5

#%%

"""
Notes:
    - run as a function of the fraction h/c. This makes is more independent.
    - the most optimal conditions for the forecast; you know the true distribution and estimate it perfectly. Then you also have a good feeling about the cost function; namely that we want to be close to zero, but on  the positive side.
"""

cost_fractions = []
unit_values = np.arange(0,150+1,1)
for c in tqdm(unit_values):
    criterion = c/(c+h)
    actual_costs_baseline = []
    actual_costs_by_hand = []
    for iteration in range(500):
        training_data, forecasts, true_time_series = generate_data(lambda_true, time_Series_length, forecast_horizon, number_of_samples)
        # Determine the actions for baseline model
        baseline_activation_time = int(N0/round(np.mean(training_data)))
        decisions_baseline = np.zeros(forecast_horizon)
        if baseline_activation_time < lead_time:
            decisions_baseline[baseline_activation_time-1:] = round(np.mean(training_data))
        else:
            decisions_baseline[lead_time-1:] = round(np.mean(training_data))
        # Determine the actions for optimal decision rule
        decisions_by_hand = np.zeros(forecast_horizon)
        for t in range(lead_time, forecast_horizon+1):
            N_t = forecasts[:,:t].sum(axis = 1) - decisions_by_hand[:t-1].sum()
            decisions_by_hand[t-1] = max(round(np.quantile(N_t, criterion))-N0, 0)
        # Determine actual costs given decision rules
        N_unknown_without_decisions = N0-np.cumsum(true_time_series[-forecast_horizon:])
        N_baseline = N_unknown_without_decisions+np.cumsum(decisions_baseline)
        actual_costs_baseline.append(actual_costs(N_baseline, h, c))
        N_by_hand = N_unknown_without_decisions+np.cumsum(decisions_by_hand)
        actual_costs_by_hand.append(actual_costs(N_by_hand, h, c))

    cost_fractions.append(np.sum(actual_costs_baseline)/np.sum(actual_costs_by_hand))

#%%

plt.figure(figsize=(8, 6))
plt.plot((h/unit_values), cost_fractions, color='blue', linewidth=0.1, marker='.', markersize=1, label="Cost Fraction")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.xlabel("h/c", fontsize=12, labelpad=10)
plt.ylabel("Lower bound on total cost increase", fontsize=12, labelpad=10)
plt.title("Cost Fractions vs. Unit Values", fontsize=14, pad=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()

plt.figure(figsize=(8, 6))
plt.plot(np.log((h/unit_values)), np.log(cost_fractions), color='blue', linewidth=0.2, marker='.', markersize=1, label="Cost Fraction")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.xlabel("ln(h/c)", fontsize=12, labelpad=10)
plt.ylabel("ln(Lower bound on total cost increase)", fontsize=12, labelpad=10)
plt.title("Cost Fractions vs. Unit Values", fontsize=14, pad=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()

#%%

unit_values = np.arange(0,150+1,1)
c = 100
criterion = c/(c+h)

actual_costs_baseline = []
actual_costs_by_hand = []
for iteration in range(500):
    training_data, forecasts, true_time_series = generate_data(lambda_true, time_Series_length, forecast_horizon, number_of_samples)
    # Determine the actions for baseline model
    baseline_activation_time = int(N0/round(np.mean(training_data)))
    decisions_baseline = np.zeros(forecast_horizon)
    if baseline_activation_time < lead_time:
        decisions_baseline[baseline_activation_time-1:] = round(np.mean(training_data))
    else:
        decisions_baseline[lead_time-1:] = round(np.mean(training_data))
    # Determine the actions for optimal decision rule
    decisions_by_hand = np.zeros(forecast_horizon)
    for t in range(lead_time, forecast_horizon+1):
        N_t = forecasts[:,:t].sum(axis = 1) - decisions_by_hand[:t-1].sum()
        decisions_by_hand[t-1] = max(round(np.quantile(N_t, criterion))-N0, 0)
    # Determine actual costs given decision rules
    N_unknown_without_decisions = N0-np.cumsum(true_time_series[-forecast_horizon:])
    N_baseline = N_unknown_without_decisions+np.cumsum(decisions_baseline)
    actual_costs_baseline.append(actual_costs(N_baseline, h, c))
    N_by_hand = N_unknown_without_decisions+np.cumsum(decisions_by_hand)
    actual_costs_by_hand.append(actual_costs(N_by_hand, h, c))


print(np.sum(actual_costs_baseline)/np.sum(actual_costs_by_hand))
# %%
