"""
Created on Tue Nov 19 21:02:40 2024

@author: Jonas Petersen
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaincc

def gamma_ratio(x,y):
    return gammaincc(x, y)

def generate_data(lambda_true, lambda_true_period, time_Series_length, forecast_horizon, number_of_samples):
    t = np.arange(time_Series_length)
    lambda_true = (lambda_true*(np.sin(2*np.pi*t/lambda_true_period)+np.sin(2*np.pi*t/(lambda_true_period*1.5))+1/2)+np.random.normal(np.sqrt(lambda_true),np.sqrt(lambda_true))).clip(0)
    true_time_series = np.random.poisson(lambda_true)
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



lambda_true = 3  # Mean (λ) of the Poisson distribution
lambda_true_period = 20
time_Series_length = 300
forecast_horizon = 52
number_of_samples = 100000

training_data, forecasts, true_time_series = generate_data(lambda_true, lambda_true_period, time_Series_length, forecast_horizon, number_of_samples)

plt.figure()
plt.plot(np.arange(time_Series_length-forecast_horizon),training_data)
plt.plot(np.arange(time_Series_length),lambda_true*(np.sin(2*np.pi*np.arange(time_Series_length)/lambda_true_period)+1)+np.random.normal(0,1))
#%%
U = np.array([0., 0., 0., 0., 0., 0., 0., 1., 3., 3., 5., 4., 4., 4., 3., 3., 5.,
       5., 2., 2., 5., 4., 1., 5., 5., 3., 3., 6., 2., 5., 2., 3., 3., 3.,
       2., 4., 3., 3., 5., 3., 3., 5., 3., 4., 4., 4., 4., 2., 1., 5., 2.,
       4.])
h = 1
c = 100
N0 = 37
gamma = np.ones(len(U))


Nt = N0 + np.cumsum(U)-np.cumsum(forecasts, axis=1)

h_term = relu(Nt)*h
c_term = relu(-Nt)*c
Cost = np.einsum("ij,j -> i", h_term+c_term, gamma)
EC = np.mean(Cost, axis = 0)

print(EC)

plt.figure()
plt.hist(Cost, bins = 50)

m = 1.8
lambdat = np.cumsum(forecasts, axis=1).mean(axis = 0)
X = N0+np.cumsum(U)
w = 1/2

e1 = np.exp(-m*(X-lambdat+w)/np.sqrt(lambdat))
e2 = np.exp(-m*(X-lambdat+w-1)/np.sqrt(lambdat))

q1 = X*1/(e1+1)
q2 = lambdat*1/(e2+1)
q3 = X-lambdat
C2 = (h+c)*(q1-q2)-c*q3
EC2 = sum(np.einsum("j,j -> j", C2, gamma))
print(EC2)


t1 = X*gamma_ratio(X+1,lambdat)
t2 = lambdat*gamma_ratio(X,lambdat)
t3 = q3
C3 = (h+c)*(t1-t2)-c*t3
EC3 = sum(np.einsum("j,j -> j", C3, gamma))
print(EC3)

#%%

plt.figure()
plt.plot(np.einsum("j,j -> j", C3, gamma))