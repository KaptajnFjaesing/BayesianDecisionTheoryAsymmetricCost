"""
Created on Wed Oct 23 08:22:21 2024

@author: Jonas Petersen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import dual_annealing

def swish(x, b):
    # Compute the exponential in a numerically stable way
    # np.clip is used to limit the values of b*x to avoid overflow
    z = np.clip(b * x, -500, 500)  # Clip to prevent overflow in exp
    return (x / (1 + np.exp(-z))).round(3)

def cost_function(U, forecasts, k_sc, k_uv, N0, gamma):
    b = 50
    Nt = N0 + np.cumsum(U)-np.cumsum(forecasts, axis=1)
    return np.dot(swish(Nt,b)*k_sc+swish(-Nt,b)*k_uv,gamma)

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
ITE = 19

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


#%% Different approaches
from scipy.optimize import root_scalar
import pandas as pd
import time

k_sc = 5
k_uv = 20
step = 1
possible_actions = np.arange(0,20)
t_forecast_horizon = np.arange(1, forecast_horizon+1)
N0 = 37

gamma_0 = 1
gamma = gamma_0**t_forecast_horizon
tic = time.time()

decisions_baseline = np.zeros(forecast_horizon)
decisions_baseline[lead_time-1:] = int(np.mean(training_data))
toc = time.time()
EC_baseline = expected_cost(decisions_baseline, forecasts, k_sc, k_uv, N0, gamma)

print("Expected cost baseline: ", EC_baseline, "; computation time", toc-tic)

tic = time.time()
possible_decisions = np.arange(0,20)
decisions_by_hand = np.zeros(forecast_horizon)
for t in range(lead_time, forecast_horizon+1):
    s_t = forecasts[:,t-1]
    N_t = N0 + possible_decisions[:,None] - forecasts[:,:t].sum(axis = 1)+decisions_by_hand[:t-1].sum()
    condition = (N_t >= 0).astype(int)
    gradient = ((k_sc + k_uv*s_t) * condition - k_uv*s_t).mean(axis = 1)
    sign_change_idx = np.where((gradient[:-1] < 0) & (gradient[1:] > 0))[0]
    if len(sign_change_idx) > 0:
        decisions_by_hand[t-1] = possible_decisions[sign_change_idx[0]+1]
    else:
        decisions_by_hand[t-1] = possible_decisions[np.argmin(gradient)]
toc = time.time()
EX_by_hand = expected_cost(decisions_by_hand, forecasts, k_sc, k_uv, N0, gamma)
print("Expected cost by hand: ", EX_by_hand, "; computation time", toc-tic)


def derivative(U, forecasts, t, N_old, k_sc, k_uv):
    s_t = forecasts[:,t-1]
    N_t = N_old+U - forecasts[:,:t].sum(axis = 1)
    condition = (N_t >= 0).astype(int)
    return ((k_sc + k_uv * s_t) * condition - k_uv * s_t).mean()

tic = time.time()
decisions_root_scalar = np.zeros(forecast_horizon)
for t in range(lead_time,forecast_horizon+1):
    try:
        decisions_root_scalar[t-1] = root_scalar(derivative, args=(forecasts, t, N0+decisions_root_scalar[:t-1].sum(), k_sc, k_uv), bracket= [0, 20]).root
    except:
        decisions_root_scalar[t-1] = 0

toc = time.time()
EC_root_scalar = expected_cost(decisions_root_scalar, forecasts, k_sc, k_uv, N0, gamma)
print("Expected cost root_scalar: ", EC_root_scalar, "; computation time", toc-tic)

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
print("Expected cost RW: ", EC_RW, "; computation time", toc-tic)

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
print("Expected cost dual_annealing:", EC_dual_annealing,  "; computation time", toc-tic)


df_decisions = pd.DataFrame(data = list(zip(decisions_baseline,decisions_by_hand,decisions_root_scalar,decisions_RW,decisions_dual_annealing)), columns = ["Baseline","By hand","root_scalar", "Numerical", "dual_annealing"])

#%%

# Actual costs
b = 50
N_baseline = N0+np.cumsum(decisions_baseline)-np.cumsum(true_time_series[-forecast_horizon:])
                                                                                                             
print("Baseline costs: ", (swish(N_baseline,b)*k_sc+swish(-N_baseline,b)*k_uv).sum())
print("By hand costs: ", k_sc*(N0+np.cumsum(decisions_by_hand)-np.cumsum(true_time_series[-forecast_horizon:])).sum())
print("Root scalar costs: ", k_sc*(N0+np.cumsum(decisions_root_scalar)-np.cumsum(true_time_series[-forecast_horizon:])).sum())
print("RW costs: ", k_sc*(N0+np.cumsum(decisions_RW)-np.cumsum(true_time_series[-forecast_horizon:])).sum())
print("Dual annealing costs: ", k_sc*(N0+np.cumsum(decisions_dual_annealing)-np.cumsum(true_time_series[-forecast_horizon:])).sum())


#%% Sanity check: Expected cost compared to real costs

decisions_baseline = np.zeros(forecast_horizon)
decisions_baseline[lead_time-1:] = int(np.mean(training_data))
toc = time.time()
EC_baseline = expected_cost(decisions_baseline, forecasts, k_sc, k_uv, N0, gamma)

gamma_0 = 1
gamma = gamma_0**t_forecast_horizon

U = decisions_baseline
b = 50
Nt = N0 + np.cumsum(U)-np.cumsum(forecasts, axis=1)
EC_cost = np.dot(swish(Nt,b)*k_sc+swish(-Nt,b)*forecasts*k_uv,gamma).mean()

print(EC_cost)
#%% Why is there a difference between by hand and root scalar?


"""
Note:
    The challenge is that there are several equivalent minima for a single
    time step. These are not equivalent when looking globally over time. 
    The root scalar makes an interpolation to find the one that is optimal,
    and this appears to yield the global optimum as well.

"""

possible_decisions = np.arange(0,20,0.25)

t = lead_time+1

s_t = forecasts[:,t-1]
N_t = N0 + possible_decisions[:,None] - forecasts[:,:t].sum(axis = 1)+U[:t-1].sum()
condition = (N_t > 0).astype(int)
abs_gradient = np.abs(((k_sc + k_uv*s_t) * condition - k_uv*s_t).mean(axis = 1))

minima = np.where(abs_gradient == min(abs_gradient))[0]

if len(minima) > 1:
    left_gradient = abs_gradient[min(minima)-1]
    right_gradient = abs_gradient[max(minima)+1]
    
    if left_gradient < right_gradient:
        dec = possible_decisions[min(minima)]
    else:
        dec = possible_decisions[max(minima)]
    
print(dec)
print(possible_decisions[minima[-1]])




#%%
As = np.arange(2,4,0.1)
print([derivative(A, forecasts, t, N0, k_sc, k_uv) for A in As])
print(As[np.argmin(np.abs([derivative(A, forecasts, t, N0, k_sc, k_uv) for A in As]))])
print(root_scalar(derivative, args=(forecasts, t, N0, k_sc, k_uv), bracket= [0, 1000]).root)


#%%


tic = time.time()
possible_decisions = np.arange(0,20)
U = np.zeros(forecast_horizon)
for t in range(lead_time, forecast_horizon+1):
    s_t = forecasts[:,t-1]
    N_t = N0 + possible_decisions[:,None] - forecasts[:,:t].sum(axis = 1)+U[:t-1].sum()
    condition = (N_t >= 0).astype(int)
    gradient = ((k_sc + k_uv*s_t) * condition - k_uv*s_t).mean(axis = 1)
    sign_change_idx = np.where((gradient[:-1] < 0) & (gradient[1:] > 0))[0][0]+1
    U[t-1] = possible_decisions[sign_change_idx]
toc = time.time()
EC_analytical1 = expected_cost(U, forecasts, k_sc, k_uv, N0, gamma)
print("Expected cost by hand: ", EC_analytical1, "; computation time", toc-tic)


def derivative(U, forecasts, t, N_old, k_sc, k_uv):
    s_t = forecasts[:,t-1]
    N_t = N_old+U - forecasts[:,:t].sum(axis = 1)
    condition = (N_t >= 0).astype(int)
    return ((k_sc + k_uv * s_t) * condition - k_uv * s_t).mean()

tic = time.time()
decisions3 = np.zeros(forecast_horizon)
for t in range(lead_time,forecast_horizon+1):
    decisions3[t-1] = root_scalar(derivative, args=(forecasts, t, N0+decisions3[:t-1].sum(), k_sc, k_uv), bracket= [0, 20]).root

toc = time.time()
EC_analytical2 = expected_cost(decisions3, forecasts, k_sc, k_uv, N0, gamma)

print("Expected cost root_scalar: ", EC_analytical2, "; computation time", toc-tic)

#%%

t = 9
Up = 0

s_t = forecasts[:,t-1]
N_t = N0 + possible_decisions[:,None] - forecasts[:,:t].sum(axis = 1)+Up
condition = (N_t >= 0).astype(int)
gradient = ((k_sc + k_uv*s_t) * condition - k_uv*s_t).mean(axis = 1)
abs_gradient = np.abs(gradient)
minima = np.where(abs_gradient == min(abs_gradient))[0]

if len(minima) > 1:
    left_gradient = abs_gradient[min(minima)-1]
    right_gradient = abs_gradient[max(minima)+1]
    if left_gradient < right_gradient:
        U[t-1] = possible_decisions[min(minima)]
    else:
        U[t-1] = possible_decisions[max(minima)]

print(root_scalar(derivative, args=(forecasts, t, N0+Up, k_sc, k_uv), bracket= [0, 1000]).root)

#%%


#%%


def calculate_gradient(
        forecasts,
        possible_decisions,
        U,
        t,
        k_sc,
        k_uv
        ):
    s_t = forecasts[:,t-1]
    N_t = N0 + possible_decisions[:,None] - forecasts[:,:t].sum(axis = 1)+U[:t-1].sum()
    condition = (N_t > 0).astype(int)
    return ((k_sc + k_uv*s_t) * condition - k_uv*s_t).mean(axis = 1)



t = lead_time

U1 = np.zeros(forecast_horizon)
U2 = np.zeros(forecast_horizon)
U2[t-1] = 1

hest = calculate_gradient(
        forecasts,
        possible_decisions,
        U1,
        t,
        k_sc,
        k_uv
        )

hest2 = calculate_gradient(
        forecasts,
        possible_decisions,
        U2,
        t,
        k_sc,
        k_uv
        )

print(all(hest == hest2))


c1 = cost_function(U1, forecasts, k_sc, k_uv, N0, gamma).mean()
c2 = cost_function(U2, forecasts, k_sc, k_uv, N0, gamma).mean()

print(c1 == c2)