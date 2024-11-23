"""
Created on Sat Nov 23 19:13:26 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaincc

np.random.seed(42)

def generate_data(lambda_true, lambda_true_period, time_Series_length, forecast_horizon, number_of_samples):
    t = np.arange(time_Series_length)
    lambda_true = (lambda_true*(np.sin(2*np.pi*t/lambda_true_period)+np.sin(2*np.pi*t/(lambda_true_period*1.5))+1/2)+np.random.normal(np.sqrt(lambda_true),np.sqrt(lambda_true))).clip(0)
    true_time_series = np.random.poisson(lambda_true)
    training_data = true_time_series[:-forecast_horizon]
    lambda_est = (np.mean(training_data)*(np.sin(2*np.pi*t[-forecast_horizon:]/lambda_true_period)+np.sin(2*np.pi*t[-forecast_horizon:]/(lambda_true_period*1.5))+1/2)).clip(0)
    forecasts_model = np.random.poisson(lambda_est, (number_of_samples,forecast_horizon))
    return training_data, forecasts_model, true_time_series

def gamma_ratio(x,y):
    return gammaincc(x, y)

def expected_cost(
        N0: int,
        h: float,
        c: float,
        U: np.array,
        forecasts: np.array,
        gamma: np.array
        ) -> float:
    lambdat = np.cumsum(forecasts, axis=1).mean(axis = 0)
    X = N0+np.cumsum(U)
    t1 = X*gamma_ratio(X+1,lambdat)
    t2 = lambdat*gamma_ratio(X,lambdat)
    t3 = X-lambdat
    return  sum(np.einsum("j,j -> j", (h+c)*(t1-t2)-c*t3, gamma))

def compute_per(
        forecast_horizon: int,
        forecasts: np.array,
        r: np.array,
        lead_time: int,
        N0: int,
        R: int
        ):
    h = 1
    c = h*r
    gamma_0 = 1
    gammat = gamma_0**np.arange(forecast_horizon)
    U_bas = np.zeros(forecast_horizon)
    U_opt = np.zeros(forecast_horizon)
    for t in range(forecast_horizon):
        zeta_t = forecasts[:,:t+1].sum(axis = 1)
        lambda_t =  zeta_t.mean()
        if t >= lead_time-1:
            U_bas[t] = max(round(lambda_t+R-U_bas[:t].sum()-N0),0)
            U_opt[t] = max(round(np.quantile(zeta_t, 1/(1+h/c))-U_opt[:t].sum()-N0), 0)
        else:
            U_bas[t] = 0
            U_opt[t] = 0
    per = expected_cost(N0 = N0, h = h, c = c, U = U_opt, forecasts = forecasts, gamma = gammat)/expected_cost(N0 = N0, h = h, c = c, U = U_bas, forecasts = forecasts, gamma = gammat)

    return per

lambda_true = 10
lambda_true_period = 10
time_Series_length = 300
forecast_horizon = 52
number_of_samples = 1000

training_data, forecasts, true_time_series = generate_data(lambda_true, lambda_true_period, time_Series_length, forecast_horizon, number_of_samples)

#%%


# Example data (replace with your actual data)
# training_data, forecasts, true_time_series = generate_data(...)

# Time points
train_length = time_Series_length - forecast_horizon
forecast_time = np.arange(train_length, time_Series_length)

# Prediction intervals
point_forecast = np.mean(forecasts, axis=0)
interval_50 = np.percentile(forecasts, [25, 75], axis=0)  # 50% interval
interval_90 = np.percentile(forecasts, [5, 95], axis=0)   # 90% interval

plt.figure(figsize=(10, 6))
plt.plot(np.arange(train_length), training_data, label='Training Data', color='blue')
plt.plot(forecast_time, true_time_series[-forecast_horizon:], label='True Time Series', color='black')
plt.fill_between(
    forecast_time, 
    interval_90[0], 
    interval_90[1], 
    color='red', 
    alpha=0.2, 
    label='90% Prediction Interval'
)
plt.fill_between(
    forecast_time, 
    interval_50[0], 
    interval_50[1], 
    color='red', 
    alpha=0.4, 
    label='50% Prediction Interval'
)
plt.xlabel('Time')
plt.ylabel('Natures decision (s)')
plt.title('Forecast with Prediction Intervals')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('./docs/SequentialDecisionMaking/figures/time_series.pdf')



#%%

N0 = 37
lead_time = 6

# Define holding costs and unit values
step1 = 0.5
step2 = 0.5
ch_ratio = np.arange(step1, 100+step1, step1)
reorder_point = np.arange(step2, 200+step2, step2)

# Calculate heatmap values

data = [
    [compute_per(
            forecast_horizon = forecast_horizon,
            forecasts = forecasts,
            r = r,
            lead_time = lead_time,
            N0 = N0,
            R = R
            ) for r in ch_ratio] for R in reorder_point
]

#%%


from tqdm import tqdm

# Calculate the total number of iterations
total_iterations = len(reorder_point) * len(ch_ratio)

# Initialize a single tqdm progress bar
with tqdm(total=total_iterations, desc="Processing") as pbar:
    data = []
    for R in reorder_point:
        row = []
        for r in ch_ratio:
            result = compute_per(
                forecast_horizon=forecast_horizon,
                forecasts=forecasts,
                r=r,
                lead_time=lead_time,
                N0=N0,
                R=R
            )
            row.append(result)
            pbar.update(1)  # Update progress bar for each iteration
        data.append(row)

#%%

def plot_per(ch_ratio,reorder_point,per):
    plt.figure(figsize=(10, 8))
    plt.imshow(per, cmap="Spectral", aspect="auto", origin="lower")
    cbar = plt.colorbar(shrink=0.8)
    custom_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cbar.set_ticks(custom_ticks)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in custom_ticks])
    cbar.set_label(r"$\frac{\mathbb{E}[C|D,I]|_{\pi=\pi^*}}{\mathbb{E}[C|D,I]|_{\pi=\pi'}}$")
    plt.xticks(ticks=np.arange(len(ch_ratio), step = 10), labels = ch_ratio[::10].astype(int))
    plt.yticks(ticks=np.arange(len(reorder_point), step = 20), labels = reorder_point[::20].astype(int))
    plt.xlabel(r"$c/h$")
    plt.ylabel(r"$R$ (reorder point)")
    plt.tight_layout()
    plt.savefig('./docs/SequentialDecisionMaking/figures/per.pdf')


PER1 = np.array([
    [data[R][r] for r in range(len(ch_ratio))] for R in range(len(reorder_point))
])

plot_per(ch_ratio,reorder_point,PER1)


#%%

plt.figure()
plt.plot(np.arange(0,100,0.1),np.sqrt(np.arange(0,100,0.1)))
plt.plot(np.arange(0,100,0.1),5*np.log(np.arange(0,100,0.1)))
plt.ylim([0,100])

