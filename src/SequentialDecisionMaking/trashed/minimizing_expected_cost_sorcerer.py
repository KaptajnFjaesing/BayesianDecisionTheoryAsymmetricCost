"""
Created on Thu Oct 10 20:38:43 2024

@author: Jonas Petersen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sorcerer.sorcerer_model import SorcererModel


# Number of time points
n = 100  
x = pd.date_range(start="2020-01-01", periods=n, freq='D')  # Create datetime index

# Linear trend parameters
true_slope = 0.01
true_intercept = 3

# Seasonality parameters
seasonal_period_1 = 30  # yearly seasonality (every 30 time steps)
seasonal_period_2 = 10  # quarterly seasonality (every 10 time steps)

# Generate seasonal components
seasonal_1 = 1.5 * np.sin(2 * np.pi * np.arange(n) / seasonal_period_1)  # Yearly seasonality
seasonal_2 = 0.8 * np.sin(2 * np.pi * np.arange(n) / seasonal_period_2)   # Quarterly seasonality

# Combine components
noise = np.random.normal(0, 0.3, size=n)  # Gaussian noise
y = true_slope * np.arange(n) + true_intercept + seasonal_1 + seasonal_2 + noise

# Create a Pandas DataFrame
df = pd.DataFrame({'date': x, 'value': y})

# Train-test split
train_size = int(0.8 * n)  # 80% for training
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

#%%
# Plot the data
plt.figure(figsize = (20,10))
plt.plot(train_df['date'], train_df['value'], label="Training Data")
plt.plot(test_df['date'], test_df['value'], label="Test Data", linestyle='dashed')
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

#%%
forecast_horizon = 26

training_data = df.iloc[:-forecast_horizon]
test_data = df.iloc[-forecast_horizon:]
# %%

model_name = "SorcererModel"
model_version = "v0.4.2"

# Sorcerer
sampler_config = {
    "draws": 2000,
    "tune": 500,
    "chains": 1,
    "cores": 1,
    "sampler": "NUTS",
    "verbose": True,
    "nuts_sampler": "numpyro"
}

number_of_weeks_in_a_year = 30

model_config = {
    "number_of_individual_trend_changepoints": int(len(training_data)/number_of_weeks_in_a_year),
    "delta_mu_prior": 0,
    "delta_b_prior": 0.1,
    "m_sigma_prior": 0.2,
    "k_sigma_prior": 0.2,
    "fourier_mu_prior": 0,
    "fourier_sigma_prior" : 1,
    "precision_target_distribution_prior_alpha": 50,
    "precision_target_distribution_prior_beta": 0.1,
    "single_scale_mu_prior": 0,
    "single_scale_sigma_prior": 1,
    "shared_scale_mu_prior": 1,
    "shared_scale_sigma_prior": 1,
    "individual_fourier_terms": [
        {'seasonality_period_baseline': number_of_weeks_in_a_year,'number_of_fourier_components': 20}
    ],
    "shared_fourier_terms": []
}

sorcerer = SorcererModel(
    model_config = model_config,
    model_name = model_name,
    model_version = model_version
    )
sorcerer.fit(
    training_data = training_data,
    sampler_config = sampler_config
    )
model_preds = sorcerer.sample_posterior_predictive(test_data = test_data)


"""
model_preds["predictions"] ~ (n_chains, n_samples, n_test, n_time_series)
"""
#%% Plot forecast along with test data
import arviz as az

time_series_column_group = ['value']

(X_train, Y_train, X_test, Y_test) = sorcerer.normalize_data(
        training_data,
        test_data
        )

hdi_values = az.hdi(model_preds)["predictions"].transpose("hdi", ...)

i = 0

fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
ax.plot(X_train, Y_train[Y_train.columns[i]], color = 'tab:red',  label='Training Data')
ax.plot(X_test, Y_test[Y_test.columns[i]], color = 'black',  label='Test Data')
ax.plot(X_test, (model_preds["predictions"].mean(("chain", "draw")).T)[i], color = 'tab:blue', label='Model')
ax.fill_between(
    X_test,
    hdi_values[0].values[:,i],
    hdi_values[1].values[:,i],
    color= 'blue',
    alpha=0.4
)
ax.set_title(time_series_column_group[i])
ax.set_xlabel('Date')
ax.set_ylabel('Values')
ax.grid(True)
ax.legend()
ax.set_xlim([-0.05,max(X_test)+0.1])

plt.savefig('./docs/report/figures/forecast.png')

# %%

N0 = 50
H = 52
t = np.arange(1,forecast_horizon+1)
psi = 1000*(forecast_horizon-t)
k = 5

y_training_min = training_data[time_series_column_group].min()
y_training_max = training_data[time_series_column_group].max()
forecasts_unnormalized = np.einsum("ij, m -> ij", model_preds["predictions"].mean('chain')[:,:,0],y_training_max-y_training_min)+y_training_min.values

#%%
import matplotlib.pyplot as plt

def determin_actions(
        current_stock_level,
        investigated_stock_levels,
        forecasts,
        stock_cost_per_time_step,
        penalty_of_stockout
        ):
    forecast_horizon = forecasts_unnormalized.shape[1]
    H =  forecast_horizon+20
    t = np.arange(1,forecast_horizon+1)
    s_cumsum = forecasts.cumsum(axis = 1)
    P_no_stockout_all_stock_levels = np.mean((s_cumsum-current_stock_level)[:,:,np.newaxis] <= investigated_stock_levels, axis = 0)
    optimal_probability_of_no_stockout = penalty_of_stockout/(penalty_of_stockout+(H-t)*stock_cost_per_time_step)
    index = np.abs(P_no_stockout_all_stock_levels.T-optimal_probability_of_no_stockout).argmin(axis = 0)
    actions = np.diff(investigated_stock_levels[index], prepend = [investigated_stock_levels[index][0]]).clip(0)
    return actions, P_no_stockout_all_stock_levels, optimal_probability_of_no_stockout

actions, P_no_stockout_all_stock_levels, optimal_probability_of_no_stockout = determin_actions(
        current_stock_level = N0,
        investigated_stock_levels = np.arange(0,100),
        forecasts = forecasts_unnormalized,
        stock_cost_per_time_step = k,
        penalty_of_stockout = psi
        )

X, Y = np.meshgrid(np.arange(forecast_horizon),np.arange(0,100))

plt.figure(figsize=(8, 6))
mesh = plt.pcolormesh(X, Y, P_no_stockout_all_stock_levels.T, shading='auto', cmap='jet')
plt.scatter(np.arange(forecast_horizon),np.cumsum(actions), color='black', label='Sum of Optimal Actions', zorder=2)
plt.colorbar(mesh, label='Probability of no stockout')
plt.ylabel('Total addition to stock level')
plt.xlabel('Forecast horizon')
plt.legend()
plt.savefig('./docs/report/figures/decisions.png')

# %%

from scipy.optimize import dual_annealing

def swish(x, b):
    # Compute the exponential in a numerically stable way
    # np.clip is used to limit the values of b*x to avoid overflow
    z = np.clip(b * x, -500, 500)  # Clip to prevent overflow in exp
    return (x / (1 + np.exp(-z))).round(3)


def cost_function(U, forecasts, H, h, t, k, psi, N0):
    b = 50
    Nt = N0 + np.cumsum(U)-np.cumsum(forecasts, axis=1)
    return k*np.dot(swish(Nt,b),H-t)+np.dot(swish(-Nt,b),psi)

def expected_cost(U, forecasts, H, h, t, k, psi, N0):
    return np.mean(cost_function(
        U = U,
        forecasts = forecasts,
        H = H,
        h = h,
        t = t,
        k = k,
        psi = psi,
        N0 = N0
    ))



bounds = [(0, 10) for _ in range(forecast_horizon)]
additional_args = (forecasts_unnormalized, H, forecast_horizon, t, k, psi, N0)
result = dual_annealing(
    expected_cost,
    bounds = bounds,
    args = additional_args,
    maxiter = 1000
)

optimal_U = result.x.round(2)
print("Optimal Actions:", optimal_U)
print("Minimum Expected Cost:", result.fun)




#%%
# Check the expected cost of a known action set
U_test = actions
U_test2 = optimal_U.round().astype(int)
expected_cost_check = np.mean(cost_function(
    U=U_test,
    forecasts=forecasts_unnormalized,
    H=H,
    h=forecast_horizon,
    t=t,
    k=k,
    psi=psi,
    N0=N0
))

print(f"Expected Cost for {U_test2}:", expected_cost_check)




#%% DEVELOPMENT 

forecasts_unnormalized2 = np.einsum("i, m -> i", model_preds["predictions"].mean(('chain', 'draw')).T.values[0],y_training_max-y_training_min)+y_training_min.values


print(N0-np.cumsum(forecasts_unnormalized2))
# %%
m = 10
stock_level = np.arange(0,100)
sum_first_m = forecasts_unnormalized[:,:forecast_horizon-m].sum(axis = 1)

P_test = np.array([np.mean(sum_first_m - N0 <= u, axis = 0) for u in stock_level])

s_cumsum = forecasts_unnormalized.cumsum(axis = 1)
P = np.mean((s_cumsum-N0)[:,:,np.newaxis] <= stock_level, axis = 0)

print(P_test)
print(P[forecast_horizon-1-m])
