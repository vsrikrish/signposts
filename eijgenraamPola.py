from __future__ import division

import math
import bisect
import numbers
import operator
import functools
import numpy as np
import scipy.stats as stats
from rhodium import *

ring = 15
max_failure_probability = 0.0005

def pola(t,                 # the current time in years
         a      = 0,        # sea level anomaly at t=0 [m]
         b      = 0.00356,  # initial rate at t=0      [m/a]
         c      = 1.19e-5,  # acceleration             [m/a^2]
         c_star = 0.01724,  # abrupt increase of rate  [m/a]
         t_star = 55):      # timing of abrupt rate increase
    return 100*(a + b*t + c*(np.power(t,2)) + c_star * ((np.sign(t - t_star) + 1) / 2) * (t - t_star))

def pola_params(file, nsamples=20000):
    content = np.loadtxt(file)
    sampled_rows = np.random.choice(content.shape[0], nsamples, replace=True)
    
    result = np.zeros((nsamples, 5))
    result[:,1:5] = content[sampled_rows, 2:6]
    result[:,(1, 2, 4)] /= 1000.0
    result[:,3] -= 2000.0

    return result
        
POLA_PARAMS = pola_params("pola_array_beta.txt", nsamples=20000)
    
def exponential_investment_cost(u,            # increase in dike height
                                h0,           # original height of the dike
                                c=125.6422,   # constant from Table 1
                                b=1.1268,     # constant from Table 1
                                lam=0.0098):  # constant from Table 1
    if u == 0:
        return 0
    else:
        return (c + b*u)*math.exp(lam*(h0+u))
    
def eijgenraam(Xs,                # list if dike heightenings
               Ts,                # time of dike heightenings
               T = 300,           # planning horizon
               P0 = 0.00137,      # constant from Table 1
               V0 = 11810.4,      # constant from Table 1
               alpha = 0.0502,    # constant from Table 1
               delta = 0.04,      # discount rate, mentioned in Section 2.2
               gamma = 0.035,     # paper says this is taken from government report, but no indication of actual value
               rho = 0.015,       # risk-free rate, mentioned in Section 2.2
               zeta = 0.003764,   # constant from Table 1
               c=125.6422,        # constant from Table 1
               b=1.1268,          # constant from Table 1
               lam=0.0098,        # constant from Table 1
               ax = 0,             # sea level anomaly at t=0 [m]
               bx = 0.00356,       # initial rate at t=0      [m/a]
               cx = 1.19e-5,       # acceleration             [m/a^2]
               c_star = 0.01724,  # abrupt increase of rate  [m/a]
               t_star = 55):      # timing of abrupt rate increase
    Ts = [int(Ts[i] + sum(Ts[:i])) for i in range(len(Ts)) if Ts[i] + sum(Ts[:i]) < T]
    Xs = Xs[:len(Ts)]
     
    if len(Ts) == 0:
        Ts = [0]
        Xs = [0]
    
    def find_height(t):
        if t < Ts[0]:
            return 0
        elif t > Ts[-1]:
            return sum(Xs)
        else:
            return sum(Xs[:bisect.bisect_right(Ts, t)])
        
    failure_probability = [P0*np.exp(alpha*pola(t, ax, bx, cx, c_star, t_star))*np.exp(-alpha*find_height(t)) for t in range(T+1)]
    failure_loss = [V0*np.exp((gamma-rho)*t)*np.exp(zeta*find_height(t)) for t in range(T+1)]
    expected_investment = np.mean([exponential_investment_cost(Xs[i], np.sum(Xs[:i]), c, b, lam)*np.exp(-delta*Ts[i]) for i in range(len(Xs))])
    expected_loss = np.mean(np.sum([failure_loss[t]*failure_probability[t] for t in range(T+1)], axis=0))
    expected_cost = expected_investment + expected_loss
    mean_failure = np.mean(failure_probability, axis=0)
    max_failure = np.max(failure_probability, axis=0)
    total_failure_probability = 1 - np.product(1 - mean_failure)
    average_failure_probability = np.mean(mean_failure)
    maximum_failure_probability = np.max(max_failure)
    return (expected_investment, expected_loss, expected_cost, total_failure_probability, average_failure_probability, maximum_failure_probability)

def eijgenraamRobust(
               Xs,                # list if dike heightenings
               Ts,                # time of dike heightenings
               T = 300,           # planning horizon
               P0 = 0.00137,      # constant from Table 1
               V0 = 11810.4,      # constant from Table 1
               alpha = 0.0502,    # constant from Table 1
               delta = 0.04,      # discount rate, mentioned in Section 2.2
               gamma = 0.035,     # paper says this is taken from government report, but no indication of actual value
               rho = 0.015,       # risk-free rate, mentioned in Section 2.2
               zeta = 0.003764,   # constant from Table 1
               c=125.6422,        # constant from Table 1
               b=1.1268,          # constant from Table 1
               lam=0.0098):       # constant from Table 1
    return eijgenraam(Xs, Ts, T, P0, V0, alpha, delta, gamma, rho, zeta, c, b, lam, 
                     ax=POLA_PARAMS[:,0], bx=POLA_PARAMS[:,1], cx=POLA_PARAMS[:,2],
                     c_star=POLA_PARAMS[:,4], t_star=POLA_PARAMS[:,3])

def plot_details(
               Xs,                # list if dike heightenings
               Ts,                # time of dike heightenings
               T = 300,           # planning horizon
               P0 = 0.00137,      # constant from Table 1
               alpha = 0.0502,    # constant from Table 1
               a = 0,             # sea level anomaly at t=0 [m]
               b = 0.00356,       # initial rate at t=0      [m/a]
               c = 1.19e-5,       # acceleration             [m/a^2]
               c_star = 0.01724,  # abrupt increase of rate  [m/a]
               t_star = 55,       # timing of abrupt rate increase
               plot_args = {}):   
    Ts = [int(Ts[i] + sum(Ts[:i])) for i in range(len(Ts)) if Ts[i] + sum(Ts[:i]) < T]
    Xs = Xs[:len(Ts)]
    
    if len(Ts) == 0:
        Ts = [0]
        Xs = [0]
        
    # convert inputs to numpy arrays
    P0 = np.asarray([P0]) if isinstance(P0, numbers.Number) else np.asarray(P0)
    a = np.asarray([a]) if isinstance(a, numbers.Number) else np.asarray(a)
    b = np.asarray([b]) if isinstance(b, numbers.Number) else np.asarray(b)
    c = np.asarray([c]) if isinstance(c, numbers.Number) else np.asarray(c)
    c_star = np.asarray([c_star]) if isinstance(c_star, numbers.Number) else np.asarray(c_star)
    t_star = np.asarray([t_star]) if isinstance(t_star, numbers.Number) else np.asarray(t_star)
    n = max([x.shape[0] for x in [P0, a, b, c, c_star, t_star]])
    
    # compute the failure probability
    def find_height(t):
        if t < Ts[0]:
            return 0
        elif t > Ts[-1]:
            return sum(Xs)
        else:
            return sum(Xs[:bisect.bisect_right(Ts, t)])
        
    failure_probability = np.zeros((n, T+1))
    
    for t in range(T+1):
        failure_probability[:,t] = P0*np.exp(alpha*pola(t, a, b, c, c_star, t_star))*np.exp(-alpha*find_height(t))

    # generate the plot
    fig = plt.figure()

    for i in range(failure_probability.shape[0]):
        plt.plot(range(T+1), failure_probability[i,:], 'b-', **plot_args)
            
    plt.plot(range(T+1), [max_failure_probability]*(T+1), 'r--')
        
    for i in range(len(Ts)):
        if Ts[i] == 0:
            plt.text(0, np.max(failure_probability[:,Ts[0]])/2, str(round(Xs[i], 1)) + " cm", ha='left', va='center')
        else:
            plt.text(Ts[i], (np.max(failure_probability[:,Ts[i]-1])+np.max(failure_probability[:,Ts[i]]))/2, str(round(Xs[i], 1)) + " cm", ha='left', va='center')
        
    if n == 1:
        plt.legend(["Failure Probability", "Current Safety Standard"])
    else:
        plt.ylim([0, 0.001])
        
    plt.xlabel("Time (years)")
    plt.ylabel("Failure Probability")
    plt.show()

model = Model(eijgenraamRobust)
 
model.parameters = [Parameter("Xs"),
                    Parameter("Ts"),
                    Parameter("T"),
                    Parameter("P0"),
                    Parameter("V0"),
                    Parameter("alpha"),
                    Parameter("delta"),
                    Parameter("gamma"),
                    Parameter("rho"),
                    Parameter("zeta"),
                    Parameter("c"),
                    Parameter("b"),
                    Parameter("lam"),
                    Parameter("ax"),
                    Parameter("bx"),
                    Parameter("cx"),
                    Parameter("c_star"),
                    Parameter("t_star")]
 
model.responses = [Response("TotalInvestment", Response.MINIMIZE),
                   Response("TotalLoss", Response.MINIMIZE),
                   Response("TotalCost", Response.INFO),
                   Response("TotalFailureProb", Response.INFO),
                   Response("AvgFailureProb", Response.MINIMIZE),
                   Response("MaxFailureProb", Response.INFO)]

model.constraints = [Constraint("TotalInvestment <= 10000.0"),
                     Constraint("TotalLoss <= 50000.0"),
                     Constraint("MaxFailureProb <= " + str(max_failure_probability))]
 
model.levers = [RealLever("Xs", 0.0, 500.0, length=10),
                RealLever("Ts", 0, 300, length=10)]

setup_cache(file="rhodium_pola.cache")
output = cache("output_pola", lambda: optimize(model, "BorgMOEA", 20000, epsilons=[10, 10, 0.00001]))

##==============================================================================
## Show a plot comparing the SLR from PoLA versus Eijgenraam's constant
##------------------------------------------------------------------------------
# plt.figure()
# plt.plot(range(301), [pola(t,
#                            a=POLA_PARAMS[:,0],
#                            b=POLA_PARAMS[:,1], 
#                            c=POLA_PARAMS[:,2], 
#                            c_star=POLA_PARAMS[:,4],
#                            t_star=POLA_PARAMS[:,3]) for t in range(301)], 'b-', alpha=0.002)
# plt.plot(range(301), [0.76*t for t in range(301)], 'r--')
# plt.xlabel("Time (years)")
# plt.ylabel("Sea Level Rise (cm)")
# plt.show()

##==============================================================================
## Show 3d scatter plot
##------------------------------------------------------------------------------
scatter3d(model, output, x="TotalInvestment", y="TotalLoss", z="MaxFailureProb",
          cmap=plt.get_cmap("rainbow"), #norm=mpl.colors.LogNorm(),
          depthshade=False, interactive=False,
          pick_handler=lambda i : plot_details(output[i]["Xs"], output[i]["Ts"],
                                               a=POLA_PARAMS[:,0],
                                               b=POLA_PARAMS[:,1], 
                                               c=POLA_PARAMS[:,2], 
                                               c_star=POLA_PARAMS[:,4],
                                               t_star=POLA_PARAMS[:,3],
                                               plot_args={"alpha" : 0.02}))
plt.show()

##==============================================================================
## Create rotating animation of 3D scatter plot
##------------------------------------------------------------------------------
# scatter3d(model, output, x="TotalInvestment", y="TotalLoss", z="MaxFailureProb",
#           cmap=plt.get_cmap("rainbow"),
#           depthshade=False)
# animate3d("eijgenraamPola", steps=360, transform=(1, 0, 0), duration=0.02)

##==============================================================================
## Plot Eijgenraam's optimal solution
##------------------------------------------------------------------------------
# plot_details([54.72, 54.72, 54.72, 54.72, 54.72, 54.72], [0, 50, 53, 53, 53, 53])
# plt.show()

##==============================================================================
## Zoom in on subset
##------------------------------------------------------------------------------
# brushed_output = find(output, "MaxFailureProb <= 0.0005")
# scatter3d(model, brushed_output, x="TotalInvestment", y="TotalLoss", z="MaxFailureProb",
#           cmap=plt.get_cmap("rainbow"),
#           depthshade=False, interactive=False,
#           pick_handler=lambda i : plot_details(output[i]["Xs"], output[i]["Ts"]))
# plt.show()

##==============================================================================
## Plot minimum cost, minimum failure probability policies
##------------------------------------------------------------------------------
# policy = find_min(output, "TotalCost")
# plot_details(policy["Xs"], policy["Ts"])
#  
# policy = find_min(output, "MaxFailureProb")
# plot_details(policy["Xs"], policy["Ts"])
