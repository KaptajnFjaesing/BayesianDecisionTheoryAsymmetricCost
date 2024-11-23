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
number_of_samples = 1000
N0 = 37

#%%

costs_baseline = []
costs_by_hand = []
holding_costs = np.arange(0,20+1,1)
unit_values = np.arange(0,100+1,1)
costs_baseline = np.zeros((len(unit_values),len(holding_costs)))
costs_by_hand = np.zeros((len(unit_values),len(holding_costs)))

for c in tqdm(range(1,len(unit_values))):
    for h in range(1,len(holding_costs)):
        criterion = unit_values[c]/(unit_values[c]+holding_costs[h])
        actual_costs_baseline = []
        actual_costs_by_hand = []
        for iteration in range(2000):
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
                decisions_by_hand[t-1] = max(round(np.quantile(N_t-N0, criterion)), 0)
            # Determine actual costs given decision rules
            N_unknown_without_decisions = N0-np.cumsum(true_time_series[-forecast_horizon:])
            N_baseline = N_unknown_without_decisions+np.cumsum(decisions_baseline)
            actual_costs_baseline.append(actual_costs(N_baseline, holding_costs[h], unit_values[c]))
            N_by_hand = N_unknown_without_decisions+np.cumsum(decisions_by_hand)
            actual_costs_by_hand.append(actual_costs(N_by_hand, holding_costs[h], unit_values[c]))
        costs_baseline[c,h] = np.sum(actual_costs_baseline)
        costs_by_hand[c,h] = np.sum(actual_costs_by_hand)
    



#%%

# Calculate heatmap values
heatmap_values = costs_by_hand / costs_baseline

# Plotting the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(heatmap_values, cmap="coolwarm", aspect="auto", origin="lower", interpolation="bilinear")
cbar = plt.colorbar(label=r"$\frac{\mathbb{E}[C^*|D,I]}{\mathbb{E}[C^{(1)}|D,I]}$", shrink=0.8)
#plt.plot(holding_costs,)
plt.xticks(ticks=np.arange(len(holding_costs)), labels=holding_costs)
plt.yticks(ticks=np.arange(0, len(unit_values), step=5), labels=unit_values[::5])
plt.xlabel("Holding Cost")
plt.ylabel("Unit Value")
plt.show()
plt.tight_layout()
plt.savefig('./docs/SequentialDecisionMaking/figures/heatmap.pdf')
#%%

constant_ratio = 0.1
indices = [(i, j) for i in range(1,len(unit_values)) for j in range(len(holding_costs)) if j / i == constant_ratio]
print(indices)
selected_elements = np.array([heatmap_values[i, j] for (i, j) in indices])
print(selected_elements)


#%%
number_of_samples = 1000
costs_baseline = []
costs_by_hand = []
holding_costs = np.arange(0,10+1,1)
unit_values = np.arange(0,50+1,1)
costs_baseline = np.zeros((len(unit_values),len(holding_costs)))
costs_by_hand = np.zeros((len(unit_values),len(holding_costs)))

for c in tqdm(range(len(unit_values))):
    h = 0
    if (unit_values[c] != 0) or (holding_costs[h] != 0):
        criterion = unit_values[c]/(unit_values[c]+holding_costs[h])
        actual_costs_baseline = []
        actual_costs_by_hand = []
        for iteration in range(100):
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
                decisions_by_hand[t-1] = max(round(np.quantile(N_t-N0, criterion)), 0)
            # Determine actual costs given decision rules
            N_unknown_without_decisions = N0-np.cumsum(true_time_series[-forecast_horizon:])
            N_baseline = N_unknown_without_decisions+np.cumsum(decisions_baseline)
            actual_costs_baseline.append(actual_costs(N_baseline, holding_costs[h], unit_values[c]))
            N_by_hand = N_unknown_without_decisions+np.cumsum(decisions_by_hand)
            actual_costs_by_hand.append(actual_costs(N_by_hand, holding_costs[h], unit_values[c]))

        costs_baseline[c,h] = np.sum(actual_costs_baseline)
        costs_by_hand[c,h] = np.sum(actual_costs_by_hand)

#%%



plt.figure(figsize=(8, 6))
plt.plot((h/unit_values), costs_baseline/costs_by_hand, color='blue', linewidth=0.1, marker='.', markersize=1, label="Cost Fraction")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.xlabel("h/c", fontsize=12, labelpad=10)
plt.ylabel("Lower bound on total cost increase", fontsize=12, labelpad=10)
plt.title("Cost Fractions vs. Unit Values", fontsize=14, pad=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()

plt.figure(figsize=(8, 6))
plt.plot(np.log((h/unit_values)), np.log(costs_baseline/costs_by_hand), color='blue', linewidth=0.2, marker='.', markersize=1, label="Cost Fraction")
plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.xlabel("ln(h/c)", fontsize=12, labelpad=10)
plt.ylabel("ln(Lower bound on total cost increase)", fontsize=12, labelpad=10)
plt.title("Cost Fractions vs. Unit Values", fontsize=14, pad=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()


#%%

plt.figure(figsize=(8, 6))
plt.plot(h/unit_values, costs_baseline, color='blue', linewidth=0.2, marker='.', markersize=1, label="Cost Fraction baseline")
#plt.plot(np.log(h/unit_values), np.log(costs_by_hand[-1]/costs_by_hand), color='red', linewidth=0.2, marker='.', markersize=1, label="Cost Fraction by hand")

plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
plt.xlabel("ln(h/c)", fontsize=12, labelpad=10)
plt.ylabel("ln(Lower bound on total cost increase)", fontsize=12, labelpad=10)
plt.title("Cost Fractions vs. Unit Values", fontsize=14, pad=15)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.legend(fontsize=10)
plt.tight_layout()


#%%

plt.figure(figsize=(8, 6))
plt.plot(np.log(h/unit_values), np.log(costs_baseline/costs_by_hand), color='blue', linewidth=0.2, marker='.', markersize=1, label="Cost Fraction")
plt.plot(np.log(h/unit_values), np.log(2.1*(h/unit_values)**(1/4)), color='red', linewidth=0.2, marker='.', markersize=1, label="Cost Fraction")

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
