"""
Created on Wed Oct 16 09:09:32 2024

@author: Jonas Petersen
"""
import numpy as np
import matplotlib.pyplot as plt

def swish(x, b):
    # Compute the exponential in a numerically stable way
    # np.clip is used to limit the values of b*x to avoid overflow
    z = np.clip(b * x, -500, 500)  # Clip to prevent overflow in exp
    return (x / (1 + np.exp(-z))).round(3)


def cost_function(U, forecasts, H, t, k, psi, N0):
    b = 50
    Nt = N0 + U-np.cumsum(forecasts, axis=1)[:,np.newaxis,:]
    return k*np.dot(swish(Nt,b),H-t)+np.dot(swish(-Nt,b),psi)

def expected_cost(U, forecasts, H, t, k, psi, N0):
    return np.mean(cost_function(
        U = U,
        forecasts = forecasts,
        H = H,
        t = t,
        k = k,
        psi = psi,
        N0 = N0
    ), axis = 0)


lambda_true = 3  # Mean (λ) of the Poisson distribution

time_Series_length = 100
forecast_horizon = 20
H = 20
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

forecast_dates = forecast_dates[:lead_time]
forecasts = forecasts_model[:,:lead_time]

median_forecasts = np.median(forecasts, axis = 0)

plt.figure(figsize = (20,10))
plt.plot(training_dates,training_data, label = "Training Data")
plt.plot(forecast_dates,true_time_series[-forecast_horizon:-forecast_horizon+lead_time], label = "Test Data")
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
plt.savefig('./docs/report/figures/forecast_example.png')
#%%
t = np.arange(1,lead_time+1)
k = 5
psi = 100*np.ones(len(t))
N0 = lead_time*lambda_true

possible_actions = np.arange(0,20)
U = np.zeros((len(possible_actions),lead_time))
U[:,-1] = possible_actions

b = 50
Nt = N0 + U-np.cumsum(forecasts, axis=1)[:,np.newaxis,:]

EC = expected_cost(U, forecasts, H, t, k, psi, N0)

# Create the plot
plt.figure(figsize=(20, 10))  # Set figure size
plt.plot(possible_actions, EC, color='blue', linewidth=2, label='Cost Function')  # Add line width and color
plt.xlabel('Possible Decisions', fontsize=14)  # X-axis label
plt.ylabel('Cost', fontsize=14)  # Y-axis label

# Customize ticks and grid
plt.xticks(fontsize=12)  # X-axis ticks font size
plt.yticks(fontsize=12)  # Y-axis ticks font size
plt.grid(color='gray', linestyle='--', alpha=0.7)  # Add a grid for better readability

plt.xticks(possible_actions, fontsize=12)  # Set x-ticks from 0 to 10 with a step of 1
plt.yticks([])  # Clear y-ticks

# Add vertical lines
plt.axvline(U[np.argmin(EC),-1], color='red', linestyle='--', linewidth=2, label='Decision that minimize cost')  # Vertical line for minimum
plt.axvline(np.sum(median_forecasts)-N0, color='orange', linestyle='--', linewidth=2, label='Decision which balance sum of point forecasts')  # Another vertical line
plt.axvline(true_time_series[-forecast_horizon:-forecast_horizon+lead_time].sum()-N0, color='brown', linestyle='--', linewidth=2, label='Perfect decision in retrospective')  # Another vertical line


# Add a legend
plt.legend(fontsize=12)

# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.savefig('./docs/report/figures/cost_function.png')

plt.figure(figsize=(20, 10))
plt.hist(np.cumsum(forecasts, axis=1)[:,-1], density = True)
plt.xlabel('Demand', fontsize=14)  # X-axis label
plt.ylabel('Probability density', fontsize=14)  # Y-axis label
plt.grid(color='gray', linestyle='--', alpha=0.7)  # Add a grid for better readability
plt.xticks(fontsize=12)  # X-axis ticks font size
plt.yticks(fontsize=12)  # Y-axis ticks font size
plt.savefig('./docs/report/figures/demand_density.png')

#%%

criterion = psi[lead_time-1]/(psi[lead_time-1]+(H-lead_time)*k)

print(np.percentile(forecasts.sum(axis = 1), 100*criterion)-N0)
