#%%
from joblib import Parallel, delayed
import numpy as np
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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

# Define a function for a single calculation to enable parallel processing
def calculate_costs(unit_val, holding_cost, criterion, lead_time, forecast_horizon, N0, ensemble_size):
    actual_costs_baseline = []
    actual_costs_by_hand = []
    
    # Perform 2000 iterations for this combination of unit_val and holding_cost
    for _ in range(ensemble_size):
        training_data, forecasts, true_time_series = generate_data(lambda_true, time_Series_length, forecast_horizon, number_of_samples)
        
        # Baseline decisions
        baseline_activation_time = int(N0 / round(np.mean(training_data)))
        decisions_baseline = np.zeros(forecast_horizon)
        if baseline_activation_time < lead_time:
            decisions_baseline[baseline_activation_time-1:] = round(np.mean(training_data))
        else:
            decisions_baseline[lead_time-1:] = round(np.mean(training_data))
        
        # Decisions by hand
        decisions_by_hand = np.zeros(forecast_horizon)
        for t in range(lead_time, forecast_horizon+1):
            N_t = forecasts[:, :t].sum(axis=1) - decisions_by_hand[:t-1].sum()
            decisions_by_hand[t-1] = max(round(np.quantile(N_t - N0, criterion)), 0)
        
        # Calculate actual costs
        N_unknown_without_decisions = N0 - np.cumsum(true_time_series[-forecast_horizon:])
        N_baseline = N_unknown_without_decisions + np.cumsum(decisions_baseline)
        actual_costs_baseline.append(actual_costs(N_baseline, holding_cost, unit_val))
        
        N_by_hand = N_unknown_without_decisions + np.cumsum(decisions_by_hand)
        actual_costs_by_hand.append(actual_costs(N_by_hand, holding_cost, unit_val))
    
    # Sum costs for the given iteration
    return np.sum(actual_costs_baseline), np.sum(actual_costs_by_hand)

lambda_true = 3  # Mean (λ) of the Poisson distribution
possible_decisions = np.arange(0,20)
time_Series_length = 300
forecast_horizon = 52
lead_time = 6
number_of_samples = 5000
ensemble_size = 5000
N0 = 37

#%%
# Preallocate arrays
holding_costs = np.arange(1, 20+1, 1)
unit_values = np.arange(1, 100+1, 1)
costs_baseline = np.zeros((len(unit_values), len(holding_costs)))
costs_by_hand = np.zeros((len(unit_values), len(holding_costs)))

# Iterate over holding_costs and unit_values using parallel processing
for h in tqdm(range(1, len(holding_costs))):
    results = Parallel(n_jobs=-1)(
        delayed(calculate_costs)(unit_values[c], holding_costs[h], unit_values[c] / (unit_values[c] + holding_costs[h]), lead_time, forecast_horizon, N0, ensemble_size)
        for c in range(1, len(unit_values))
    )
    
    for c, (cost_baseline, cost_by_hand) in enumerate(results, start=1):
        costs_baseline[c, h] = cost_baseline
        costs_by_hand[c, h] = cost_by_hand

# %%


# Calculate heatmap values
heatmap_values = costs_by_hand / costs_baseline

# Plotting the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_values, cmap="coolwarm", aspect="auto", origin="lower", interpolation="bilinear")
cbar = plt.colorbar(label=r"$\frac{\mathbb{E}[C^*|D,I]}{\mathbb{E}[C^{(1)}|D,I]}$", shrink=0.8)
#plt.plot(holding_costs,)
plt.xticks(ticks=np.arange(len(holding_costs)), labels=holding_costs)
plt.yticks(ticks=np.arange(len(unit_values), step=5), labels=unit_values[::5])
plt.xlabel("Holding Cost")
plt.ylabel("Unit Value")
plt.tight_layout()
plt.show()

#%%

import pickle

# After the computation loop is complete
# Save costs_baseline
with open("costs_baseline.pkl", "wb") as f:
    pickle.dump(costs_baseline, f)

# Save costs_by_hand
with open("costs_by_hand.pkl", "wb") as f:
    pickle.dump(costs_by_hand, f)

# %%
