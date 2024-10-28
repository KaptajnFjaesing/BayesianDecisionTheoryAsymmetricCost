"""
Created on Thu Oct 10 20:38:43 2024

@author: Jonas Petersen
"""

import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100  # number of time points
x = np.arange(n)

# Linear trend parameters
true_slope = 0.1
true_intercept = 1.0

# Seasonality parameters
seasonal_period_1 = 30  # yearly seasonality (every 30 time steps)
seasonal_period_2 = 10  # quarterly seasonality (every 10 time steps)

# Generate seasonal components
seasonal_1 = 1.5 * np.sin(2 * np.pi * x / seasonal_period_1)  # Yearly seasonality
seasonal_2 = 0.8 * np.sin(2 * np.pi * x / seasonal_period_2)   # Quarterly seasonality

# Combine components
noise = np.random.normal(0, 0.3, size=n)  # Gaussian noise
y = true_slope * x + true_intercept + seasonal_1 + seasonal_2 + noise

# Plot the dummy time series with seasonalities
plt.plot(x, y, label="Observed Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Dummy Time Series with Seasonalities")
plt.legend()
plt.show()

# %%
with pm.Model() as model:
    X = pm.Data('input', x, dims = 'obs')
    Y = pm.Data('observations', y)
    slope = pm.Normal("slope", mu=0, sigma=1)
    intercept = pm.Normal("intercept", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=1)
    mu = slope * X + intercept
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=Y, dims = 'obs')


# %%
with model:
    trace = pm.sample(draws = 200, tune = 100, cores = 1, chains = 1, return_inferencedata=True)

#%%
future_x = np.arange(n, n + 10)
with model:
    pm.set_data({"input": future_x})
    posterior_predictive = pm.sample_posterior_predictive(trace)
#%%
import arviz as az


forecast_y = posterior_predictive.posterior_predictive["y_obs"].mean(('draw','chain'))
hdi_values = az.hdi(posterior_predictive.posterior_predictive)["y_obs"].transpose("hdi", ...)


plt.figure()
plt.plot(x, y, label="Observed Data")
plt.plot(future_x, forecast_y, label="Forecast", color="red")
plt.fill_between(
    future_x,
    hdi_values[0].values,
    hdi_values[1].values,
    color= 'blue',
    alpha = 0.4
)
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Dummy Time Series Forecast")
plt.legend()
plt.show()




# %%

t = np.arange(8,15,0.1)
thres = []
probabilities = []
sum_forecasts = posterior_predictive.posterior_predictive["y_obs"].sum('obs')
for i in t:
    threshold = len(future_x)*i
    probabilities.append((sum_forecasts <= threshold).mean(('draw','chain')))
    thres.append(threshold)

plt.figure()
plt.plot(thres, probabilities)


# %%

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

h = len(future_x)

N0 = 100
H = 52
t = np.arange(1,h+1)
psi = 1000*np.ones(h)
k = 5

forecasts = posterior_predictive.posterior_predictive["y_obs"].mean('chain') #(n_samples, n_test)


# %%

from scipy.optimize import dual_annealing


bounds = [(0, 20) for _ in range(h)]
additional_args = (forecasts, H, h, t, k, psi, N0)
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
U_test = [0, 0, 0, 0, 0, 0, 0, 0, 5, 17]
expected_cost_check = np.mean(cost_function(
    U=U_test,
    forecasts=forecasts,
    H=H,
    h=h,
    t=t,
    k=k,
    psi=psi,
    N0=N0
))

print(f"Expected Cost for {U_test}:", expected_cost_check)

# %% 

"""
- I can iteratively determine the optimal configuration from behind?
- I need to store decisions iteratively an add them to N0.

- MAYBE this works because the function is monotonic. Perhaps with a set of
actions that is constrained by batches?



"""

for m in range(h):

    sum_first_m = posterior_predictive.posterior_predictive["y_obs"].isel(obs=slice(0, h-m)).sum('obs')
    
    U = range(0,100)
    
    P = np.array([(sum_first_m - N0 <= u).mean(('draw','chain')).values for u in U])
    index = (np.abs(P-1/(1+(H-t[m-1])*k/psi[h-m-1]))).argmin()
    #print(t[m])
    print(U[index])


#%%



LHS = ((H-t)*k+psi)*P

m = 9

print(sum(LHS[m:]))
print(sum(psi[m:]))

print()