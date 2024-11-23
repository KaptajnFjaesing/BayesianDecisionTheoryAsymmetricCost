"""
Created on Mon Nov 18 18:44:05 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaincc

def generate_data(lambda_true, lambda_true_period, time_Series_length, forecast_horizon, number_of_samples):
    t = np.arange(time_Series_length)
    lambda_true = (lambda_true*(np.sin(2*np.pi*t/lambda_true_period)+np.sin(2*np.pi*t/(lambda_true_period*1.5))+1/2)+np.random.normal(np.sqrt(lambda_true),np.sqrt(lambda_true))).clip(0)
    true_time_series = np.random.poisson(lambda_true)
    training_data = true_time_series[:-forecast_horizon]
    lambda_est = (np.mean(training_data)*(np.sin(2*np.pi*t[-forecast_horizon:]/lambda_true_period)+np.sin(2*np.pi*t[-forecast_horizon:]/(lambda_true_period*1.5))+1/2)).clip(0)
    forecasts_model = np.random.poisson(lambda_est, (number_of_samples,forecast_horizon))
    return training_data, forecasts_model, true_time_series


def generate_data2(lambda_true, lambda_true_period, time_Series_length, forecast_horizon, number_of_samples):
    true_time_series = np.random.poisson(lambda_true, time_Series_length)
    training_data = true_time_series[:-forecast_horizon]
    forecasts_model = np.random.poisson(np.mean(training_data), (number_of_samples,forecast_horizon))
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

def expected_cost_approximation(
        N0: int,
        h: float,
        c: float,
        U: np.array,
        forecasts: np.array,
        gamma: np.array
        ) -> float:
    m = 1.8
    lambdat = np.cumsum(forecasts, axis=1).mean(axis = 0)
    lambdat = np.where(lambdat == 0, 1e-10, lambdat)
    X = N0+np.cumsum(U)
    e1 = np.exp(-m*(X-lambdat+1/2)/np.sqrt(lambdat))
    e2 = np.exp(-m*(X-lambdat+1/2-1)/np.sqrt(lambdat))
    q1 = X*1/(e1+1)
    q2 = lambdat*1/(e2+1)
    q3 = X-lambdat
    return sum(np.einsum("j,j -> j", (h+c)*(q1-q2)-c*q3, gamma))


def expected_cost_iteration(
        h: float,
        c: float,
        N0: int,
        U: int,
        lambdat : int,
        gamma: float
        ) -> float:
    m = 1.8
    X = N0+U
    if lambdat > 0:
        e1 = np.exp(-m*(X-lambdat+1/2)/np.sqrt(lambdat))
        e2 = np.exp(-m*(X-lambdat+1/2-1)/np.sqrt(lambdat))
    else:
        e1 = 0
        e2 = 0
    q1 = X*1/(e1+1)
    q2 = lambdat*1/(e2+1)
    q3 = X-lambdat
    return ((h+c)*(q1-q2)-c*q3)*gamma

def E_p_approx(
        h: float,
        c: float,
        lambdat : int,
        gamma: float
        ) -> float:
    m = 1.8
    e1 = np.exp(m/np.sqrt(lambdat))
    return gamma*lambdat*h*(e1-1)/(e1*h/c+1)

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
    E_p = np.zeros(forecast_horizon)
    E_p2 = np.zeros(forecast_horizon)
    E_L = np.zeros(forecast_horizon)
    E_R = np.zeros(forecast_horizon)
    for t in range(forecast_horizon):
        zeta_t = forecasts[:,:t+1].sum(axis = 1)
        lambda_t =  zeta_t.mean()
        if t >= lead_time-1:
            U_bas[t] = max(round(lambda_t+R-U_bas[:t].sum()-N0),0)
            U_opt[t] = max(round(np.quantile(zeta_t, 1/(1+h/c))-U_opt[:t].sum()-N0), 0)
            E_p[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = U_opt[:t+1].sum(), lambdat = lambda_t, gamma = 1)
            if U_opt[t] >  0:
                E_p2[t] = E_p_approx(h = h, c = c, lambdat = lambda_t , gamma = 1)
            else:
                E_p2[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = U_opt[:t+1].sum(), lambdat = lambda_t, gamma = 1)
            E_R[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = U_bas[:t+1].sum(), lambdat = lambda_t, gamma = 1)
        else:
            U_bas[t] = 0
            U_opt[t] = 0
            E_L[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = 0, lambdat = lambda_t, gamma = 1)
    per1 = expected_cost_approximation(N0 = N0, h = h, c = c, U = U_opt, forecasts = forecasts, gamma = gammat)/expected_cost_approximation(N0 = N0, h = h, c = c, U = U_bas, forecasts = forecasts, gamma = gammat)   
    per2 = expected_cost(N0 = N0, h = h, c = c, U = U_opt, forecasts = forecasts, gamma = gammat)/expected_cost(N0 = N0, h = h, c = c, U = U_bas, forecasts = forecasts, gamma = gammat)
    per3 = (sum(E_L)+sum(E_p))/(sum(E_L)+sum(E_R))
    per4 = (sum(E_L)+sum(E_p2))/(sum(E_L)+sum(E_R))
    
    return per1, per2, U_opt, U_bas, per3, per4


lambda_true = 10
lambda_true_period = 10
time_Series_length = 300
forecast_horizon = 52
number_of_samples = 1000

training_data, forecasts, true_time_series = generate_data(lambda_true, lambda_true_period, time_Series_length, forecast_horizon, number_of_samples)


#%%

N0 = 37
lead_time = 6

# Define holding costs and unit values
step1 = 1
step2 = 5
ch_ratio = np.arange(1, 100+step1, step1)
reorder_point = np.arange(1, 200+step2, step2)

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

def plot_per(ch_ratio,reorder_point,per):
    plt.figure(figsize=(10, 8))
    plt.imshow(per, cmap="Spectral", aspect="auto", origin="lower", vmin = 0, vmax = 1.1)
    cbar = plt.colorbar(shrink=0.8)
    custom_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    cbar.set_ticks(custom_ticks)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in custom_ticks])
    cbar.set_label(r"$\frac{\mathbb{E}[C^*|D,I]}{\mathbb{E}[C^{(1)}|D,I]}$")
    plt.xticks(ticks=np.arange(len(ch_ratio), step = 10), labels = ch_ratio[::10].astype(int))
    plt.yticks(ticks=np.arange(len(reorder_point), step = 20), labels = reorder_point[::20].astype(int))
    plt.xlabel(r"$c/h$")
    plt.ylabel(r"$R$ (reorder point)")
    plt.tight_layout()
    plt.show()

PER1 = np.array([
    [data[R][r][0] for r in range(len(ch_ratio))] for R in range(len(reorder_point))
])

PER2 = np.array([
    [data[R][r][1] for r in range(len(ch_ratio))] for R in range(len(reorder_point))
])

PER3 = np.array([
    [data[R][r][4] for r in range(len(ch_ratio))] for R in range(len(reorder_point))
])

PER4 = np.array([
    [data[R][r][5] for r in range(len(ch_ratio))] for R in range(len(reorder_point))
])

plot_per(ch_ratio,reorder_point,PER1)
plot_per(ch_ratio,reorder_point,PER2)
plot_per(ch_ratio,reorder_point,PER3)
plot_per(ch_ratio,reorder_point,PER4)

#%%


r = 50
R = 50
print("r: ", r)
print("R: ", R)

hest = compute_per(
        forecast_horizon = forecast_horizon,
        forecasts = forecasts,
        r = r,
        lead_time = lead_time,
        N0 = N0,
        R = R
        )

print(hest[0],hest[1],hest[4],hest[5])

#%%

h = 1
c = h*r
gamma_0 = 1
gammat = gamma_0**np.arange(forecast_horizon)
U_bas = np.zeros(forecast_horizon)
U_opt = np.zeros(forecast_horizon)
E_p = np.zeros(forecast_horizon)
E_p2 = np.zeros(forecast_horizon)
E_L = np.zeros(forecast_horizon)
E_R = np.zeros(forecast_horizon)
for t in range(forecast_horizon):
    zeta_t = forecasts[:,:t+1].sum(axis = 1)
    lambdat =  zeta_t.mean()
    if t >= lead_time-1:
        U_bas[t] = max(round(lambdat+R-U_bas[:t].sum()-N0),0)
        U_opt[t] = max(round(np.quantile(zeta_t, 1/(1+h/c))-U_opt[:t].sum()-N0), 0)
        E_p[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = U_opt[:t+1].sum(), lambdat = lambdat, gamma = 1)
        if U_opt[t] >  0:
            E_p2[t] = E_p_approx(h = h, c = c, lambdat = lambdat , gamma = 1)
        else:
            E_p2[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = U_opt[:t+1].sum(), lambdat = lambdat, gamma = 1)
        E_R[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = U_bas[:t+1].sum(), lambdat = lambdat, gamma = 1)
    else:
        U_bas[t] = 0
        U_opt[t] = 0
        E_L[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = 0, lambdat = lambdat, gamma = 1)

ko = expected_cost_approximation(N0 = N0, h = h, c = c, U = U_opt, forecasts = forecasts, gamma = gammat)/expected_cost_approximation(N0 = N0, h = h, c = c, U = U_bas, forecasts = forecasts, gamma = gammat)
ko2 = (sum(E_L)+sum(E_p))/(sum(E_L)+sum(E_R))
ko3 = (sum(E_L)+sum(E_p2))/(sum(E_L)+sum(E_R))

print(ko)
print(ko2)
print(ko3)

#%%
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
    E_p = np.zeros(forecast_horizon)
    E_p2 = np.zeros(forecast_horizon)
    E_L = np.zeros(forecast_horizon)
    E_R = np.zeros(forecast_horizon)
    for t in range(forecast_horizon):
        lambdat =  forecasts[:,:t+1].sum(axis = 1).mean()
        if t >= lead_time-1:
            U_bas[t] = max(round(lambdat+R-U_bas[:t].sum()-N0),0)
            zeta_t = forecasts[:,:t+1].sum(axis = 1)
            U_opt[t] = max(round(np.quantile(zeta_t, 1/(1+h/c))-U_opt[:t].sum()-N0), 0)
            E_p[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = U_opt[:t+1].sum(), lambdat = lambdat, gamma = 1)
            if U_opt[t] >  0:
                E_p2[t] = E_p_approx(h = h, lambdat = lambdat , gamma = 1)
            else:
                E_p2[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = U_opt[:t+1].sum(), lambdat = lambdat, gamma = 1)
            E_R[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = U_bas[:t+1].sum(), lambdat = lambdat, gamma = 1)
        else:
            U_bas[t] = 0
            U_opt[t] = 0
            E_L[t] = expected_cost_iteration(h = h, c = c, N0 = N0, U = 0, lambdat = lambdat, gamma = 1)
    per1 = expected_cost_approximation(N0 = N0, h = h, c = c, U = U_opt, forecasts = forecasts, gamma = gammat)/expected_cost_approximation(N0 = N0, h = h, c = c, U = U_bas, forecasts = forecasts, gamma = gammat)   
    per2 = expected_cost(N0 = N0, h = h, c = c, U = U_opt, forecasts = forecasts, gamma = gammat)/expected_cost(N0 = N0, h = h, c = c, U = U_bas, forecasts = forecasts, gamma = gammat)
    per3 = (sum(E_L)+sum(E_p))/(sum(E_L)+sum(E_R))
    per4 = (sum(E_L)+sum(E_p2))/(sum(E_L)+sum(E_R))
    
    return per1, per2, U_opt, U_bas, per3, per4
