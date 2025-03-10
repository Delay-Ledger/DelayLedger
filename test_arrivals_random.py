import numpy as np
import matplotlib.pyplot as plt
import random
from utils import create_piecewise_function, expected_pv, cost_jump_arrivals
from params import *

random.seed(1)

### Set necessary parameters ###
n_plots = 9 # number of plots in figure

# Iterations in Simulation
iters = 100000

# Begin figure of n_plots
plt.figure(figsize=(n_plots*2, n_plots))
# X values
x = np.linspace(0, round_T, 500)

### Creating plots with random arrivals of cost surges ###

for p in range(n_plots):
    # v_f values for each graph flight set at random
    v_f = np.random.randint(1,10)
    # converting  dollars per minute to dollars per hour
    
    # Define piecewise breakpoints, these are the times for cost surges
    breakpoints = cost_jump_arrivals(lambda_parameter,round_T)

    # Create the piecewise function
    f = create_piecewise_function(breakpoints, slope=v_f, surge=surge)

    # Using piecewise function to find y values
    y1 = f(x*60)

    # Expected value function
    expected_value_rate = expected_pv(round_T,lambda_parameter,v_f,surge)
    e = lambda x: expected_value_rate*x
    y2 = e(x*60)

    # Regular v_f
    v = lambda x: v_f*x
    y3 = v(x*60)

    # Subplot
    plt.subplot(3, 3, p+1)
    plt.plot(x, y1, label="Piecewise Function")
    plt.plot(x, y2, label="Expected Value")
    plt.plot(x, y3, label="Flight Value")
    plt.xlabel("delay time (hours)")
    plt.ylabel("accumulated cost ($)")
    plt.legend()
    plt.title(rf'Flight {p}, $\nu_f$ = {v_f}, $\lambda$ = 1 / {lambda_parameter} hours')

plt.tight_layout()
plt.show()

### Comparing Simulation to Expected Value Calculation ###

total_value = []
m = np.random.randint(1,10)
for i in range(iters):
    # Define piecewise breakpoints, these are the times for cost surges
    breakpoints = cost_jump_arrivals(lambda_parameter,round_T)

    # Create the piecewise function
    f = create_piecewise_function(breakpoints, slope=m, surge=surge)

    total_value.append(f(round_T*60))

simulation_total_value = np.mean(total_value)
rate_total_value = np.mean(simulation_total_value)/60/round_T # to get $ / min
theory_value = expected_pv(round_T,lambda_parameter,m,surge)

# Simulated Calculation
print(f"Simulation Result | Rate: {rate_total_value:.2f} $/min")
print(f"Theory Result | Rate: {theory_value:.2f} $/min")

# ### Flight Value and Expected Flight Value ###

# # Define piecewise breakpoints, these are the times for cost surges
# breakpoints = cost_jump_arrivals(lambda_parameter,round_T)

# # Create the piecewise function
# f = create_piecewise_function(breakpoints, slope=m*60, surge=surge)
    
# # Using piecewise function to find y values
# y1 = f(x)

# # Expected value function
# e = lambda x: theory_value*60*x
# y2 = e(x)

# # Regular v_f
# r = lambda x: m*x
# y3 = r(x)

# # Plot them on the same axes
# plt.plot(x, y1, label="Piecewise")
# plt.plot(x, y2, label="Expected Value")
# plt.plot(x, y3, label="Flight Value")

# # Add a legend, labels, and show the plot
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Comparing Flight Value and Expected Value")
# plt.legend()
# plt.show()

# import os 
# import pandas as pd
# path = os.getcwd()+'/test_stochastic/intra-alt-intra/2019-05-01/baseline/intra1/df_gurobi.csv'
# df_nonstoch = pd.read_csv(path)
# plt.scatter(df_nonstoch['new_delay_15bin_weighted'])
# print(df_nonstoch)