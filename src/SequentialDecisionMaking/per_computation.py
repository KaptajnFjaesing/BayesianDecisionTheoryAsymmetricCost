"""
Created on Wed Nov 13 19:24:22 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_data(lambda_true, time_Series_length, forecast_horizon, number_of_samples):
    true_time_series = np.random.poisson(lambda_true, time_Series_length)
    training_data = true_time_series[:-forecast_horizon]
    forecasts_model = np.random.poisson(np.mean(training_data), (number_of_samples,forecast_horizon))
    return training_data, forecasts_model, true_time_series

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


def ET(
        h: float,
        c: float,
        N0: int,
        U: int,
        lambdat : int,
        gamma: float
        ):
    m = 1.8
    return gamma*((h+c)*((N0+U)/(np.exp(-m*(N0+U+1-lambdat)/np.sqrt(lambdat))+1)-lambdat/(np.exp(-m*(N0+U-lambdat)/np.sqrt(lambdat))+1))-c*(N0+U-lambdat))

def compute_per(
        forecast_horizon: int,
        forecasts: np.array,
        h: np.array,
        c: np.array,
        lead_time: int,
        N0: int,
        R: int,
        true_time_series: np.array
        ):
    gamma_0 = 1
    U_bas = np.zeros(forecast_horizon)
    U_opt = np.zeros(forecast_horizon)
    E_bas = np.zeros(forecast_horizon)
    E_opt = np.zeros(forecast_horizon)
    for t in range(forecast_horizon):
        lambdat =  forecasts[:,:t+1].sum(axis = 1).mean()
        if t >= lead_time-1:
            U_bas[t] = max(round(lambdat+R-U_bas[:t].sum()-N0),0)
            N_t = forecasts[:,:t+1].sum(axis = 1) - U_opt[:t].sum()
            U_opt[t] = max(round(np.quantile(N_t, 1/(1+h/c))-N0), 0)
        else:
            U_bas[t] = 0
            U_opt[t] = 0

        E_bas[t] = ET(
            h = h,
            c = c,
            N0 = N0,
            U = U_bas[:t].sum(),
            lambdat = lambdat,
            gamma = gamma_0**(t+1)
            )
        E_opt[t] = ET(
            h = h,
            c = c,
            N0 = N0,
            U = U_opt[:t].sum(),
            lambdat = lambdat,
            gamma = gamma_0**(t+1)
            )
    
    per1 = np.sum(E_opt)/np.sum(E_bas)
    per2 = expected_cost(U_opt, forecasts, h, c, N0, 1*np.arange(forecast_horizon))/expected_cost(U_bas, forecasts, h, c, N0, 1*np.arange(forecast_horizon))
    
    return per1, per2, U_opt, U_bas


def plot_per(per):
    plt.figure(figsize=(10, 8))
    plt.imshow(per, cmap="Spectral", aspect="auto", origin="lower")
    cbar = plt.colorbar(shrink=0.8)

    # Set custom ticks for colorbar
    custom_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cbar.set_ticks(custom_ticks)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in custom_ticks])
    cbar.set_label(r"$\frac{\mathbb{E}[C^*|D,I]}{\mathbb{E}[C^{(1)}|D,I]}$")

    # Adjust ticks to match holding costs and unit values
    plt.xticks(ticks=np.arange(len(holding_costs), step = 10), labels = holding_costs[::10].astype(int))
    plt.yticks(ticks=np.arange(len(unit_values), step = 22), labels = unit_values[::22].astype(int))

    # Labels
    plt.xlabel(r"$h_t$ (Holding Cost)")
    plt.ylabel(r"$c_t$ (Unit Value)")
    plt.tight_layout()

    # Display and save the plot
    plt.show()


lambda_true = 3  # Mean (λ) of the Poisson distribution
time_Series_length = 300
forecast_horizon = 52
number_of_samples = 1000

training_data, forecasts, true_time_series = generate_data(lambda_true, time_Series_length, forecast_horizon, number_of_samples)


#%%

N0 = 37
R = 3*10
lead_time = 6

# Define holding costs and unit values
step = 0.5
holding_costs = np.arange(1, 20+step, step)
unit_values = np.arange(1, 100+step, step)

# Calculate heatmap values

data = [
    [compute_per(
            forecast_horizon = forecast_horizon,
            forecasts = forecasts,
            h = h,
            c = c,
            lead_time = lead_time,
            N0 = N0,
            R = R,
            true_time_series = true_time_series
            ) for h in holding_costs] for c in unit_values
]

#%%
PER1 = np.array([
    [data[c][h][0] for h in range(len(holding_costs))] for c in range(len(unit_values))
])

PER2 = np.array([
    [data[c][h][1] for h in range(len(holding_costs))] for c in range(len(unit_values))
])


plot_per(PER1)
plot_per(PER2)


#%%

uopt = np.array([
    [data[c][h][2] for h in range(len(holding_costs))] for c in range(len(unit_values))
])
ubas = np.array([
    [data[c][h][3] for h in range(len(holding_costs))] for c in range(len(unit_values))
])