from __future__ import division

import time
import math
import bisect
import operator
import functools
import numpy as np
import scipy.stats as stats
from rhodium import *

def failureProbability(X,           # the increase in dike height in meters
                       p0 = 0.0038, # the failure probability when H=0
                       alpha = 2.6, # tunable factor for the failure probability
                       eta = 0.0):  # total structural sea level change (e.g., sinking dikes, sea level rise) in meters
    return p0*np.exp(-alpha*(X-eta))

def failureLoss(t,             # the current time
                V = 2e10,      # the initial value of the land, people, goods, etc.
                delta = 0.02): # the discount rate (e.g., interest rate - growth rate)
    return V*np.power(1 + delta, -t)

def investment(X,           # the increase in dike height in meters
               I0 = 0,      # the base cost incurred during a dike heightening
               k = 42.01e6): # the per-meter cost of dike heightening
    return I0 + k*X

def fixedRate(t,         # the current time in years
              eta=0.01): # the structural seal level change in meters / year
    return t*eta

def pola(t,                 # the current time in years
         a      = 0,        # sea level anomaly at t=0 [m]
         b      = 0.00356,  # initial rate at t=0      [m/a]
         c      = 1.19e-5,  # acceleration             [m/a^2]
         c_star = 0.01724,  # abrupt increase of rate  [m/a]
         t_star = 55):      # timing of abrupt rate increase
    return a + b*t + c*(np.power(t,2)) + c_star * ((np.sign(t - t_star) + 1) / 2) * (t - t_star)

def pola_params(file, nsamples=20000):
    content = np.loadtxt(file)
    sampled_rows = np.random.choice(content.shape[0], nsamples, replace=True)
    
    result = np.zeros((nsamples, 5))
    result[:,1:5] = content[sampled_rows, 2:6]
    result[:,(1, 2, 4)] /= 1000.0
    result[:,3] -= 2000.0

    return result
        
POLA_PARAMS = pola_params("pola_array_beta.txt")

def _vanDantzig(X,            # the increase in dike height in meters
               T,             # the planning horizon in years
               p0 = 0.0038,   # the failure probability when H=0
               alpha = 2.6,   # tunable factor for the failure probability
               V = 2e10,      # the initial value of the land, people, goods, etc.
               delta = 0.02,  # the discount rate (e.g., interest rate - growth rate)
               I0 = 0,        # the base cost incurred during a dike heightening
               k = 42.01e6,   # the per-meter cost of dike heightening
               slr_func = fixedRate): # function for calculating sea level at time t
    failure_probability = [failureProbability(X, p0, alpha, slr_func(t)) for t in range(T+1)]
    failure_loss = [failureLoss(t, V, delta) for t in range(T+1)]
    expected_investment = np.mean(investment(X, I0, k))
    expected_loss = np.mean(np.sum([failure_loss[t]*failure_probability[t] for t in range(T+1)], axis=0))
    expected_cost = expected_investment + expected_loss
    mean_failure = np.mean(failure_probability, axis=0)
    max_failure = np.max(failure_probability, axis=0)
    total_failure_probability = 1 - np.product(1 - mean_failure)
    average_failure_probability = np.mean(mean_failure)
    maximum_failure_probability = np.max(max_failure)
    return (expected_investment, expected_loss, expected_cost, total_failure_probability, average_failure_probability, maximum_failure_probability)
    
def vanDantzig(X,             # the increase in dike height in meters
               T,             # the planning horizon in years
               p0 = 0.0038,   # the failure probability when H=0
               alpha = 2.6,   # tunable factor for the failure probability
               V = 2e10,      # the initial value of the land, people, goods, etc.
               delta = 0.02,  # the discount rate (e.g., interest rate - growth rate)
               I0 = 0,        # the base cost incurred during a dike heightening
               k = 42.01e6,   # the per-meter cost of dike heightening
               eta = 0.01):   # the structural change in sea level height in meters / year
    return _vanDantzig(X, T, p0, alpha, V, delta, I0, k, lambda t : fixedRate(t, eta))

def vanDantzigPola(
               X,             # the increase in dike height in meters
               T,             # the planning horizon in years
               p0 = 0.0038,   # the failure probability when H=0
               alpha = 2.6,   # tunable factor for the failure probability
               V = 2e10,      # the initial value of the land, people, goods, etc.
               delta = 0.02,  # the discount rate (e.g., interest rate - growth rate)
               I0 = 0,        # the base cost incurred during a dike heightening
               k = 42.01e6,   # the per-meter cost of dike heightening
               a = 0,         # sea level anomaly at t=0 [m]
               b = 0.00356,   # initial rate at t=0      [m/a]
               c = 1.19e-5,   # acceleration             [m/a^2]
               c_star = 0.01724, # abrupt increase of rate  [m/a]
               t_star = 55):     # timing of abrupt rate increase
    return _vanDantzig(X, T, p0, alpha, V, delta, I0, k, lambda t : pola(t, a, b, c, c_star, t_star))

def vanDantzigPolaRobust(
               X,             # the increase in dike height in meters
               T,             # the planning horizon in years
               p0 = 0.0038,   # the failure probability when H=0
               alpha = 2.6,   # tunable factor for the failure probability
               V = 2e10,      # the initial value of the land, people, goods, etc.
               delta = 0.02,  # the discount rate (e.g., interest rate - growth rate)
               I0 = 0,        # the base cost incurred during a dike heightening
               k = 42.01e6):  # the per-meter cost of dike heightening
    return vanDantzigPola(X, T, p0, alpha, V, delta, I0, k, POLA_PARAMS[:,0],
                          POLA_PARAMS[:,1], POLA_PARAMS[:,2], POLA_PARAMS[:,4],
                          POLA_PARAMS[:,3])
    
def exponential_investment_cost(u,            # increase in dike height
                                h0,           # original height of the dike
                                c=125.6422,   # constant from Table 1, no 11
                                b=1.1268,     # constant from Table 1, no 11
                                lam=0.0098):  # constant from Table 1, no 11
    if u == 0:
        return 0
    else:
        return (c + b*u)*math.exp(lam*(h0+u))
    
def eijgenraam(Xs,                # list if dike heightenings
               Ts,                # time of dike heightenings
               T = 300,           # planning horizon
               P0 = 0.00137,      # constant from Table 1, no 11
               V0 = 11810.4,      # constant from Table 1, no 11
               alpha = 0.0502,    # constant from Table 1, no 11
               delta = 0.04,      # discount rate, mentioned in Section 2.2
               eta = 0.76,        # constant from Table 1, no 11
               gamma = 0.035,     # paper says this is taken from government report, but no indication of actual value
               rho = 0.015,       # risk-free rate, mentioned in Section 2.2
               zeta = 0.003764,   # constant from Table 1, no 11
               investment_cost = exponential_investment_cost):

    S0 = P0*V0
    beta = alpha*eta + gamma - rho
    theta = alpha - zeta
    
    # calculate investment
    investment = 0
    
    for i in range(len(Xs)):
        step_cost = investment_cost(Xs[i], 0 if i==0 else sum(Xs[:i]))
        step_discount = math.exp(-delta*Ts[i])
        investment += step_cost * step_discount
    
    # calculate expected losses
    losses = math.exp((beta-delta)*Ts[0]) - 1
    
    for i in range(len(Xs)-1):
        losses += math.exp(-theta*sum(Xs[:(i+1)]))*(math.exp((beta - delta)*Ts[i+1]) - math.exp((beta - delta)*Ts[i]))
        
    losses += math.exp(-theta*sum(Xs))*(math.exp((beta - delta)*T) - math.exp((beta - delta)*Ts[-1]))
    losses = losses * S0 / (beta - delta)
    
    def find_height(t):
        if t < Ts[0]:
            return 0
        elif t > Ts[-1]:
            return Xs[-1]
        else:
            return Xs[bisect.bisect_left(Ts, t)]
    
    failure_probability = [P0*np.exp(-alpha*(find_height(t)-eta)) for t in range(T+1)]
    total_failure = 1 - reduce(operator.mul, failure_probability, 1)
    mean_failure = sum(failure_probability) / (T+1)
    max_failure = max(failure_probability)
    
    return (investment, losses, investment+losses, total_failure, mean_failure, max_failure)

print eijgenraam([54.72, 54.72, 54.72, 54.72, 54.72, 54.72],
                 [0, 50, 103, 156, 209, 262])
    
def lhs(n, k, transform=None):
    lhs = np.empty((n, 6))
    d = 1.0 / n
    
    for i in range(6):
        column = np.empty(n)
        
        for j in range(n):
            column[j] = random.uniform(j*d, (j+1)*d)
            
        np.random.shuffle(column)
        lhs[:,i] = column
        
    if transform:
        lhs = transform(lhs)
        
    return lhs
    
def transform(lhs, p0=0.0038, alpha=2.6, V0=2e4, delta=0.02, k=42.01):
    lhs[:,0] = p0*stats.lognorm.ppf(lhs[:,0], 0.25)
    lhs[:,1] = stats.norm.ppf(lhs[:,1], alpha, 0.1)
    lhs[:,2] = stats.norm.ppf(lhs[:,2], V0, 1e3)
    lhs[:,3] = delta*stats.lognorm.ppf(lhs[:,3], 0.1)
    lhs[:,4] = stats.norm.ppf(lhs[:,4], k, 4.0)
    return lhs

def to_SOWs(samples, names):
    return [{k:v for k, v in zip(names, samples[i,:])} for i in range(samples.shape[0])]

model = Model(vanDantzigPolaRobust)
 
model.parameters = [Parameter("X"),
                    Parameter("T", default_value=75),
                    Parameter("p0"),
                    Parameter("alpha"),
                    Parameter("V"),
                    Parameter("delta"),
                    Parameter("I0"),
                    Parameter("k")]
 
model.responses = [Response("TotalInvestment", Response.MINIMIZE),
                   Response("TotalLoss", Response.MINIMIZE),
                   Response("TotalCost", Response.INFO),
                   Response("TotalFailureProb", Response.INFO),
                   Response("AvgFailureProb", Response.INFO),
                   Response("MaxFailureProb", Response.MINIMIZE)]
 
model.levers = [RealLever("X", 0.0, 5.0)]
 
model.uncertainties = [RealUncertainty("p0", 0.0, 0.01),
                       RealUncertainty("alpha", 2.0, 3.0),
                       RealUncertainty("V", 1e10, 10e10),
                       RealUncertainty("delta", 0.0, 0.05),
                       RealUncertainty("k", 30e6, 50e6)]
 
# output = optimize(model, "NSGAII", 500)
# scatter2d(model, output, x="TotalInvestment", y="TotalLoss", c="MaxFailureProb", cmap=plt.get_cmap("rainbow"), interactive=True)
# plt.show()

# SOWs = to_SOWs(lhs(1000, 5, transform=transform), names=model.uncertainties.keys())
# 
# policy = {"X" : 0.0}
# output = evaluate(model, fix(SOWs, policy))
# scatter2d(model, output, x="TotalInvestment", y="TotalLoss", c="MaxFailureProb", cmap=plt.get_cmap("rainbow"))
# plt.show()
# 
# policy = {"X" : 2.26}
# output = evaluate(model, fix(SOWs, policy))
# scatter2d(model, output, x="TotalInvestment", y="TotalLoss", c="MaxFailureProb", cmap=plt.get_cmap("rainbow"))
# plt.show()
# 
# policy = {"X" : 4.26}
# output = evaluate(model, fix(SOWs, policy))
# scatter2d(model, output, x="TotalInvestment", y="TotalLoss", c="MaxFailureProb", cmap=plt.get_cmap("rainbow"))
# plt.show()

# 
# 
#for i in range(300):
#    X = i / 100
#    H = 4.25 + X
#    print X, H, vanDantzigPolaRobust(X, 75)
#    #print H, exceedenceProbability(H, 75), cost(H), totalCost(H, 75)
#     