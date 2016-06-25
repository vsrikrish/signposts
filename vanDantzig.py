from __future__ import division

import time
import math
import operator
import functools
import numpy as np
import scipy as sp
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

# function: compute investment cost to increase dike heightg
def investment(X,             # the increase in dike height in meters
               t,             # the current time in years
               I0 = 0,        # the base cost incurred during a dike heightening
               k = 42.01e6,   # the per-meter cost of dike heightening
               delta = 0.02): # the discount rate (e.g., interest rate - growth rate)
    return (I0 + k*X)*pow(1 + delta, -t)


# function: compute modifier to effective dike height due to land subsdience and 
# (constant-rate) sea level rise
def fixedRate(t,          # the current time in years,
              subs=0.002, # rate of subsidence (meter/year) 
              eta=0.008): # the structural sea level change in meters / year

    return t*(eta+subs)

# function: global sea level rise for van Dantzig analysis, based on Lempert et al 2012.
def pola(t,                 # the current time in years
         a      = 0,        # sea level anomaly at t=0 [m]
         b      = 0.00356,  # initial rate at t=0      [m/a]
         c      = 1.19e-5,  # acceleration             [m/a^2]
         c_star = 0.01724,  # abrupt increase of rate  [m/a]
         t_star = 55):      # timing of abrupt rate increase

    eta = a + b*t + c*(t**2) + c_star * ((math.copysign(1, t - t_star) + 1) / 2) * (t - t_star)
    return eta

def pola_params(file, nsamples=20000):
    content = np.loadtxt(file,skiprows=1)
    sampled_rows = np.random.choice(content.shape[0], nsamples, replace=True)
    result = []
    
    for row in sampled_rows:
        entry = {}
        entry["a"] = content[row][0] / 1000.0                           # sl compared to 2000
        entry["b"] = content[row][1] / 1000.0          # mm  -> m
        entry["c"] = content[row][2] / 1000.0       # mm  -> m
        entry["t_star"] = content[row][3] - 2000.0 # yrs -> yrs from y0
        entry["c_star"] = content[row][4] / 1000.0   # mm  -> m
        result.append(entry)

    return result

POLA_PARAMS = pola_params("data/array_beta.txt")
    
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
    failure_probability = [failureProbability(sum(X[:t]), p0, alpha, fixedRate(t, subs, eta)) for t in range(T+1)]
    failure_loss = [failureLoss(t, V, delta) for t in range(T+1)]
    yearly_investment = np.array([investment(X[t-1], t, I0, k, delta) for t in range(T+1)])
    total_investment = yearly_investment.sum()
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
               I0 = 0,        # the base sublime text 3 ucost incurred during a dike heightening
               k = 42.01e6,   # the per-meter cost of dike heightening
#               subs = 0.002,  # rate of land subsidence (meter/year)
#               eta = 0.008,   # rate of sea level rise (meter/year)  
               a = 0,         # sea level anomaly at t=0 [m]
               b = 0.00356,   # initial rate at t=0      [m/a]
               c = 1.19e-5,   # acceleration             [m/a^2]
               c_star = 0.01724, # abrupt increase of rate  [m/a]
               t_star = 55):     # timing of abrupt rate increase
    failure_probability = [failureProbability(sum(X[:t]), p0, alpha, pola(t, a, b, c, c_star, t_star)) for t in range(T+1)]
    failure_loss = [failureLoss(t, V, delta) for t in range(T+1)]
    yearly_investment = np.array([investment(X[t-1], t, I0, k, delta) for t in range(T+1)])
    total_investment = yearly_investment.sum()
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
        total_failure_probability = reduce(operator.mul, [p for p in failure_probability], 1)
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
    
model = Model(vanDantzigPola)

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
                    Parameter("k"),
 #                   Parameter("subs"),
 #                   Parameter("eta"),
                    Parameter("a"),
                    Parameter("b"),
                    Parameter("c"),
                    Parameter("c_star"),
                    Parameter("t_star")]

model.responses = [Response("TotalInvestment", Response.INFO),
                   Response("TotalLoss", Response.INFO),
                   Response("TotalCost", Response.MINIMIZE),
                   Response("TotalFailureProb", Response.MINIMIZE),
                   #Response("AvgFailureProb", Response.INFO)]
                   Response("MaxFailureProb", Response.INFO)]

model.levers = [RealLever("X", 0.0, 2.0, length=75)]

# model.uncertainties = [RealUncertainty("p0", 0.0, 0.01),
#                        RealUncertainty("alpha", 2.0, 3.0),
#                        RealUncertainty("V", 1e10, 10e10),
#                        RealUncertainty("delta", 0.0, 0.05),
#                        RealUncertainty("k", 30e6, 50e6),
#                        RealUncertainty("b", 0.0, 0.01),
#                        RealUncertainty("c", 0.0, 1e-4),
#                        RealUncertainty("c_star", 0.0, 0.05),
#                        RealUncertainty("t_star", 25, 65)]

model.uncertainties = [LogNormalUncertainty("p0", math.log(0.0038), 0.25),
                       NormalUncertainty("alpha", 2.6, 0.1),
                       NormalUncertainty("V", 1e10, 1e9),
                       LogNormalUncertainty("delta", math.log(0.02), 0.1),
                       NormalUncertainty("k", 42.01e6, 4e6)]
#                       LogNormalUncertainty("subs",math.log(0.002),0.1),
 #                      LogNormalUncertainty("eta",math.log(0.008),0.1)]

SOWs = sample_lhs(model,100)

for i, SOW in enumerate(SOWs):
  SOW.update(POLA_PARAMS[random.randrange(len(POLA_PARAMS))])

output = robust_optimize(model, SOWs, "NSGAII", 1000)
output = output.sort(columns=["TotalInvestment"])
print output

sns.set()

# scatter2d(model, output, x="TotalInvestment", y="TotalFailureProb", c="TotalCost")
# plt.show()

# parallel_coordinates(model, output, c="TotalCost", colormap="rainbow")
# plt.show()

# scatter3d(model,output,x="TotalInvestment",y="MaxFailureProb",z="TotalCost",
#   c="Regret Type 1")
# plt.show()


policy = output[0]
result = evaluate(model,policy)
print "Total Investment:", result["TotalInvestment"]
print "Total Losses:", result["TotalLoss"]
print "Total Failure Prob:", result["TotalFailureProb"]
print "Max Failure Prob:", result["MaxFailureProb"]

SOWs = sample_lhs(model,10000)

for i, SOW in enumerate(SOWs):
  SOW.update(POLA_PARAMS[random.randrange(len(POLA_PARAMS))])

results = evaluate(model,update(SOWs,policy))
classification = results.apply("'Reliable' if MaxFailureProb < 0.001 else 'Unreliable'")

p = Prim(results, classification, include=model.uncertainties.keys(), coi="Reliable")
box = p.find_box()
box.show_details()
fig = box.show_tradeoff()

result = sa(model, "MaxFailureProb", policy=policy, method="sobol", nsamples=10000)
print(result)
fig = result.plot()

fig = result.plot_sobol(threshold=0.01)

fig = oat(model, "MaxFailureProb", policy=policy, nsamples=1000)

#for i in range(300):
#    X = i / 100
#    H = 4.25 + X
#    print X, H, totalCost(H, 75)
    #print H, exceedenceProbability(H, 75), cost(H), totalCost(H, 75)
    