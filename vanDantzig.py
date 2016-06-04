from __future__ import division

import time
import math
import operator
import functools
import numpy as np
from rhodium import *

# function: probability of flooding (assumes stationary flooding frequency)
def failureProbability(X,           # the increase in dike height in meters
                       p0 = 0.0038, # the failure probability when H=0
                       alpha = 2.6, # tunable factor for the failure probability
                       eta = 0.0):  # total structural sea level change (e.g., sinking dikes, sea level rise) in meters
    return p0*math.exp(-alpha*(X-eta))

# function: compute NPV of losses due to flooding in a given year
def failureLoss(t,             # the current time
                V = 2e10,      # the initial value of the land, people, goods, etc.
                delta = 0.02): # the discount rate (e.g., interest rate - growth rate)
    return V*pow(1 + delta, -t)

# function: compute investment cost to increase dike height
def investment(X,           # the increase in dike height in meters
               I0 = 0,      # the base cost incurred during a dike heightening
               k = 42.01e6): # the per-meter cost of dike heightening
    return I0 + k*X


# function: compute modifier to effective dike height due to land subsdience and 
# (constant-rate) sea level rise
def fixedRate(t,          # the current time in years,
              subs=0.002, # rate of subsidence (meter/year) 
              eta=0.008): # the structural sea level change in meters / year

    return t*(eta+subs)

def pola(t,                 # the current time in years
         a      = 0,        # sea level anomaly at t=0 [m]
         b      = 0.00356,  # initial rate at t=0      [m/a]
         c      = 1.19e-5,  # acceleration             [m/a^2]
         c_star = 0.01724,  # abrupt increase of rate  [m/a]
         t_star = 55):      # timing of abrupt rate increase
    return a + b*t + c*(t**2) + c_star * ((math.copysign(1, t - t_star) + 1) / 2) * (t - t_star)

def pola_params(file, nsamples=20000):
    content = np.loadtxt(file)
    sampled_rows = np.random.choice(content.shape[0], nsamples, replace=True)
    result = []
    
    for row in sampled_rows:
        entry = {}
        entry["init_gsl_anomaly"] = 0.0                            # sl compared to 2000
        entry["init_gsl_rate"] = content[row][2] / 1000.0          # mm  -> m
        entry["gsl_acceleration"] = content[row][3] / 1000.0       # mm  -> m
        entry["timing_abrupt_increase"] = content[row][4] - 2000.0 # yrs -> yrs from y0
        entry["abrupt_rate_increase"] = content[row][5] / 1000.0   # mm  -> m
        result.append(entry)
        
    return result
        
# POLA_PARAMS = pola_params("pola_array_beta.txt")
    
def vanDantzig(X,               # the increase in dike height in meters
               T = 75,          # the planning horizon in years
               p0 = 0.0038,     # the failure probability when H=0
               alpha = 2.6,     # tunable factor for the failure probability
               V = 2e10,        # the initial value of the land, people, goods, etc.
               delta = 0.02,    # the discount rate (e.g., interest rate - growth rate)
               I0 = 0,          # the base cost incurred during a dike heightening
               k = 42.01e6,     # the per-meter cost of dike heightening
               subs = 0.002,    # rate of land subsidence (meter/year)
               eta = 0.008):    # rate of sea level rise (meter/year)
    failure_probability = [failureProbability(X, p0, alpha, fixedRate(t, subs, eta) for t in range(T+1)]
    failure_loss = [failureLoss(t, V, delta) for t in range(T+1)]
    total_investment = investment(X, I0, k)
    total_loss = sum([failure_loss[t]*failure_probability[t] for t in range(T+1)])
    total_cost = total_investment + total_loss
    total_failure_probability = reduce(operator.mul, [1 - p for p in failure_probability], 1)
    average_failure_probability = np.average(failure_probability)
    maximum_failure_probability = np.max(failure_probability)
    return (total_investment, total_loss, total_cost, total_failure_probability, average_failure_probability, maximum_failure_probability)

def vanDantzigPola(
               X,             # the increase in dike height in meters
               T,             # the planning horizon in years
               p0 = 0.0038,   # the failure probability when H=0
               alpha = 2.6,   # tunable factor for the failure probability
               V = 2e10,      # the initial value of the land, people, goods, etc.
               delta = 0.02,  # the discount rate (e.g., interest rate - growth rate)
               I0 = 0,        # the base cost incurred during a dike heightening
               k = 42.01e6,    # the per-meter cost of dike heightening
               a = 0,         # sea level anomaly at t=0 [m]
               b = 0.00356,   # initial rate at t=0      [m/a]
               c = 1.19e-5,   # acceleration             [m/a^2]
               c_star = 0.01724, # abrupt increase of rate  [m/a]
               t_star = 55):     # timing of abrupt rate increase
    failure_probability = [failureProbability(X, p0, alpha, pola(t, a, b, c, c_star, t_star)) for t in range(T+1)]
    failure_loss = [failureLoss(t, V, delta) for t in range(T+1)]
    total_investment = investment(X, I0, k)
    total_loss = sum([failure_loss[t]*failure_probability[t] for t in range(T+1)])
    total_cost = total_investment + total_loss
    total_failure_probability = reduce(operator.mul, [1 - p for p in failure_probability], 1)
    average_failure_probability = np.average(failure_probability)
    maximum_failure_probability = np.max(failure_probability)
    return [total_investment, total_loss, total_cost, total_failure_probability, average_failure_probability, maximum_failure_probability]

def vanDantzigPolaRobust(
               X,             # the increase in dike height in meters
               T,             # the planning horizon in years
               p0 = 0.0038,   # the failure probability when H=0
               alpha = 2.6,   # tunable factor for the failure probability
               V = 2e10,      # the initial value of the land, people, goods, etc.
               delta = 0.02,  # the discount rate (e.g., interest rate - growth rate)
               I0 = 0,        # the base cost incurred during a dike heightening
               k = 42.01e6):   # the per-meter cost of dike heightening
    results = np.zeros((len(POLA_PARAMS), 6))
    
    t0 = time.time()
    for i, param in enumerate(POLA_PARAMS):
        a = param["init_gsl_anomaly"]
        b = param["init_gsl_rate"]
        c = param["gsl_acceleration"]
        c_star = param["abrupt_rate_increase"]
        t_star = param["timing_abrupt_increase"]
        failure_probability = [failureProbability(X, p0, alpha, pola(t, a, b, c, c_star, t_star)) for t in range(T+1)]
        failure_loss = [failureLoss(t, V, delta) for t in range(T+1)]
        total_investment = investment(X, I0, k)
        total_loss = sum([failure_loss[t]*failure_probability[t] for t in range(T+1)])
        total_cost = total_investment + total_loss
        total_failure_probability = reduce(operator.mul, [1 - p for p in failure_probability], 1)
        average_failure_probability = np.average(failure_probability)
        maximum_failure_probability = np.max(failure_probability)
        results[i][0] = total_investment
        results[i][1] = total_loss
        results[i][2] = total_cost
        results[i][3] = total_failure_probability
        results[i][4] = average_failure_probability
        results[i][5] = maximum_failure_probability
        #return [total_investment, total_loss, total_cost, total_failure_probability, average_failure_probability, maximum_failure_probability]
        #results[i] = vanDantzigPola(X, T, p0, alpha, V, delta, I0, k,
        #                            param["init_gsl_anomaly"],
        #                            param["init_gsl_rate"],
        #                            param["gsl_acceleration"],
        #                            param["abrupt_rate_increase"],
        #                            param["timing_abrupt_increase"])
    print time.time() - t0
        
    print "Done"
    return np.mean(results, axis=0)
    
model = Model(vanDantzig)

# model.parameters = [Parameter("X"),
#                     Parameter("T", default_value=75),
#                     Parameter("p0"),
#                     Parameter("alpha"),
#                     Parameter("V"),
#                     Parameter("delta"),
#                     Parameter("I0"),
#                     Parameter("k"),
#                     Parameter("a"),
#                     Parameter("b"),
#                     Parameter("c"),
#                     Parameter("c_star"),
#                     Parameter("t_star")]

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
                   Response("TotalCost", Response.INFO)]
                   #Response("TotalFailureProb", Response.INFO),
                   #Response("AvgFailureProb", Response.INFO),
                   #Response("MaxFailureProb", Response.MINIMIZE)]

model.levers = [RealLever("X", 0.0, 5.0)]

# model.uncertainties = [RealUncertainty("p0", 0.0, 0.01),
#                        RealUncertainty("alpha", 2.0, 3.0),
#                        RealUncertainty("V", 1e10, 10e10),
#                        RealUncertainty("delta", 0.0, 0.05),
#                        RealUncertainty("k", 30e6, 50e6),
#                        RealUncertainty("b", 0.0, 0.01),
#                        RealUncertainty("c", 0.0, 1e-4),
#                        RealUncertainty("c_star", 0.0, 0.05),
#                        RealUncertainty("t_star", 25, 65)]

model.uncertainties = [UniformUncertainty("p0", 0.0, 0.01),
                       UniformUncertainty("alpha", 2.0, 3.0),
                       UniformUncertainty("V", 1e10, 10e10),
                       UniformUncertainty("delta", 0.0, 0.05),
                       UniformUncertainty("k", 30e6, 50e6)]

output = optimize(model, "NSGAII", 500)
print output

sns.set()

scatter2d(model, output, x="TotalInvestment", y="TotalLoss", c="TotalCost")
plt.show()

parallel_coordinates(model, output, c="TotalCost", colormap="rainbow")
plt.show()


#for i in range(300):
#    X = i / 100
#    H = 4.25 + X
#    print X, H, totalCost(H, 75)
    #print H, exceedenceProbability(H, 75), cost(H), totalCost(H, 75)
    