import numpy as np
import matplotlib.pyplot as plt
import random
from params import *

random.seed(1)

def expected_pv(T, lambda_parameter, slope, surge):
    # slope is in dollars per minute, T is in hours
    # must get a dollars per minute expected value slope
    return (T * 60 * slope + surge * (1/lambda_parameter) * T)/T/60

def cost_jump_arrivals(lambda_parameter, T):
    """
    Cost jumps that form the piecewise linear cost function of flights
    arrive randomly through a poisson process. This function, determines
    the timestamps when a random cost surge may occur.

    lambda_parameter: a uniform-random-generated variable as input for poisson
    T: length of round

    Returns: an arrivals array with timestamps for when cost jumps occur
    note: the arrivals array will always have 0 in its first index
    """

    # create array for arrivals and have 0 as the first index
    arrivals = []
    arrivals.append(0) 
    # ensures that there is an index to add to
    # also ensures that piecewise conditions start at x >= 0

    # start index
    i = 0

    # create a list of arrival times that are cumulative
    # with intervals between arrivals being exponentially distributed
    while arrivals[i] < T:
        # randomly sample exponential variable with parameter lambda
        v = np.random.exponential(lambda_parameter)
        
        # we are only interested in costs within the round
        if v + arrivals[i] > T:
            break

        # add cumulative values to arrivals array
        arrivals.append(v + arrivals[i])

        #iterate
        i += 1

    return arrivals

def create_piecewise_function(breakpoints, slope, surge):
    """
    Creates a piecewise function with dynamically generated function mappings.
    The number of breakpoints are random, so a dynamic function was needed.
    Each piecewise break defines when the rate of the cost per delay changes.
    The breaks are in units of hours (since the round length is in hours),
    so we need to be careful to change these to units of minutes, as this  
    function operates in minutes.
    The values of the slopes and surge for each piecewise interval may be 
    different in the future, but they are set to the same value right now.

    This a piecewise continuous function, where each arrival prompts the 
    change of the functions slope.

    breakpoints: list of arrival times for each cost surge, in hours
    slope: the slope of the linear segments, in units of dollars per minute
    surge: the magnitude of the cost surge at each piecewise breakpoint

    Returns: a function that evaluates piecewise expressions
    """

    num_segments = len(breakpoints)
    # print(breakpoints)
    
    # tracker_f created in order to track function values during piecewise construction
    tracker_f = lambda x: slope * x  

    # List to store each sub-function as a part of the piecewise function
    function_list = []

    # Create function segments dynamically
    for i in range(num_segments):
        if i == 0:
            tracker_f = lambda x: slope * x
            function_list.append(lambda x: slope * x)  # First segment (before first breakpoint)
        else:
            # Apply cost surge at each breakpoint
            # Breakpoints are in hours, convert to minutes

            y_piecewise = tracker_f(breakpoints[i] * 60)  
            plt.plot(breakpoints[i], 0, marker='o', linestyle='None', color='red', markersize = 2)  # 'o' = dot marker

            # y_piecewise = tracker_f(breakpoints[i] * 60) + surge

            # y = ax + b --> b =  y - ax
            intercept = y_piecewise - (slope + i*surge) * breakpoints[i] * 60 # Solve for new function
            tracker_f = lambda x, intercept=intercept, i=i: (slope + i*surge) * x + intercept  # Update tracker

            # intercept = y_piecewise - slope * breakpoints[i] * 60
            # tracker_f = lambda x, intercept=intercept: slope * x + intercept #update tracker

            # Add next piecewise interval
            function_list.append(lambda x, intercept=intercept, i=i: (slope + i*surge) * x + intercept) 
            # function_list.append(lambda x, intercept=intercept: slope * x + intercept)

    # Create conditions as boolean masks to pass into np.piecewise function
    def condition_masks(x):
        masks = [(breakpoints[i]*60 <= x) & (x < breakpoints[i+1]*60) for i in range(len(breakpoints) - 1)]
        masks.append(x >= breakpoints[-1]*60)  # Last condition (x ≥ last breakpoint)
        return masks

    return lambda x: np.piecewise(x, condition_masks(x), function_list)


# ########################################################################################

# ### Set necessary parameters ###
# n_plots = 9 # number of plots in figure

# # Iterations in Simulation
# iters = 100000

# # Begin figure of n_plots
# plt.figure(figsize=(n_plots*2, n_plots))
# # X values
# x = np.linspace(0, round_T, 500)

# ### Creating plots with random arrivals of cost surges ###

# for p in range(n_plots):
#     plt.subplot(3, 3, p+1)

#     # v_f values for each graph flight set at random
#     v_f = np.random.randint(1,10)
#     # converting  dollars per minute to dollars per hour
    
#     # Define piecewise breakpoints, these are the times for cost surges
#     breakpoints = cost_jump_arrivals(lambda_parameter,round_T)
#     # print(breakpoints)

#     # Create the piecewise function
#     f = create_piecewise_function(breakpoints, slope=v_f, surge=8)

#     # Using piecewise function to find y values
#     y1 = f(x*60)

#     # # Expected value function
#     # expected_value_rate = expected_pv(round_T,lambda_parameter,v_f,surge)
#     # e = lambda x: expected_value_rate*x
#     # y2 = e(x*60)

#     # # Regular v_f
#     # v = lambda x: v_f*x
#     # y3 = v(x*60)

#     # Subplot
    
#     plt.plot(x, y1, label="Piecewise Function")
#     # plt.plot(x, y2, label="Expected Value")
#     # plt.plot(x, y3, label="Flight Value")
#     plt.xlabel("delay time (hours)")
#     plt.ylabel("accumulated cost ($)")
#     plt.legend()
#     plt.title(rf'Flight {p}, $\nu_f$ = {v_f}, $\lambda$ = 1 / {lambda_parameter} hours')

# plt.tight_layout()
# plt.show()

# ### Comparing Simulation to Expected Value Calculation ###

# # total_value = []
# # m = np.random.randint(1,10)
# # for i in range(iters):
# #     # Define piecewise breakpoints, these are the times for cost surges
# #     breakpoints = cost_jump_arrivals(lambda_parameter,round_T)

# #     # Create the piecewise function
# #     f = create_piecewise_function(breakpoints, slope=m, surge=surge)

# #     total_value.append(f(round_T*60))

# # simulation_total_value = np.mean(total_value)
# # rate_total_value = np.mean(simulation_total_value)/60/round_T # to get $ / min
# # theory_value = expected_pv(round_T,lambda_parameter,m,surge)

# # # Simulated Calculation
# # print(f"Simulation Result | Rate: {rate_total_value:.2f} $/min")
# # print(f"Theory Result | Rate: {theory_value:.2f} $/min")

# # ### Flight Value and Expected Flight Value ###

# # Define piecewise breakpoints, these are the times for cost surges
# m = 5
# s = 5

# breakpoints = cost_jump_arrivals(lambda_parameter,round_T)

# # Create the piecewise function
# f = create_piecewise_function(breakpoints, slope=m, surge=s)
    
# # Using piecewise function to find y values
# y1 = f(x*60)

# # Expected value function
# # e = lambda x: theory_value*x
# # y2 = e(x*60)

# # Regular v_f
# r = lambda x: m*x
# y3 = r(x*60)

# # Plot them on the same axes
# plt.plot(x, y1, label="Piecewise")
# # plt.plot(x, y2, label="Expected Value")
# plt.plot(x, y3, label="Flight Value")

# # Add a legend, labels, and show the plot
# plt.xlabel("delay hours")
# plt.ylabel("accumulated cost ($)")
# plt.title("Comparing Flight Value and Expected Value")
# plt.legend()
# plt.show()

# #### Plot for rate on y axis

# ex_breakpoints = [0,3.5,19.2]
# x1 = np.linspace(ex_breakpoints[0],ex_breakpoints[1],500)
# x2 = np.linspace(ex_breakpoints[1],ex_breakpoints[2],500)
# x3 = np.linspace(ex_breakpoints[2],24,500)

# f1 = lambda x: 5 * x/x
# f2 = lambda x: 10 * x/x
# f3 = lambda x: 15 * x/x

# y1 = f1(x)
# y2 = f2(x)
# y3 = f3(x)

# plt.plot(x1, y1)
# plt.plot(x2, y2)
# plt.plot(x3, y3)

# # Add a legend, labels, and show the plot
# plt.xlabel("delay hours")
# plt.ylabel("cost delay rate ($/min)")
# plt.title("Piecewise increments in rate of cost")
# plt.legend()
# plt.show()

##########################
# Multiple Lines 1 Plot
##########################
m = 5
s = 5
first = []
second = []
third = []
fourth = []
fifth = []
sixth = []
seventh = []
eighth = []
ninth = []
tenth = []

functions_mvp = []
breakpoints_mvp = []
x = np.linspace(0, round_T, 500)

for i in range(0,1000):
    breakpoints = cost_jump_arrivals(lambda_parameter,round_T)
    print(breakpoints)
    # add all breakpoints for the mvp function
    for j in range(len(breakpoints)):
        breakpoints_mvp.append(breakpoints[j])

    # first element is always 0
    first.append(breakpoints[1]) if len(breakpoints) >= 2 else ""
    second.append(breakpoints[2]) if len(breakpoints) >= 3 else ""
    third.append(breakpoints[3]) if len(breakpoints) >= 4 else ""
    fourth.append(breakpoints[4]) if len(breakpoints) >= 5 else ""
    fifth.append(breakpoints[5]) if len(breakpoints) >= 6 else ""
    sixth.append(breakpoints[6]) if len(breakpoints) >= 7 else ""
    seventh.append(breakpoints[7]) if len(breakpoints) >= 8 else ""
    eighth.append(breakpoints[8]) if len(breakpoints) >= 9 else ""
    ninth.append(breakpoints[9]) if len(breakpoints) >= 10 else ""
    tenth.append(breakpoints[10]) if len(breakpoints) >= 11 else ""

    # Create the piecewise function
    f = create_piecewise_function(breakpoints, slope=m, surge=s)
    # add all functions to the mvp function
    functions_mvp.append(f)
        
    # Using piecewise function to find y values
    # y1 = f(x*60)

    # Expected value function
    # e = lambda x: theory_value*x
    # y2 = e(x*60)

    # Regular v_f
    # r = lambda x: m*x
    # y3 = r(x*60)

    # Plot them on the same axes
    # plt.plot(x, y1)
    # plt.plot(x, y2, label="Expected Value")
    # plt.plot(x, y3, label="Flight Value")

print("Average First Arrival",np.mean(first))
print("Average Second Arrival",np.mean(second))
print("Average Third Arrival",np.mean(third))
print("Average Fourth Arrival",np.mean(fourth))
print("Average Fifth Arrival",np.mean(fifth))
print("Average Sixth Arrival",np.mean(sixth))
print("Average Seventh Arrival",np.mean(seventh))
print("Average Eigth Arrival",np.mean(eighth))
print("Average Ninth Arrival",np.mean(ninth))
print("Average Tenth Arrival",np.mean(tenth))

########################
# MVP Calculation
########################

clean_breakpoints_mvp = sorted(set(breakpoints_mvp))
print("Number of Breakpoints:", len(clean_breakpoints_mvp))
print("Number of Function Samples:", len(functions_mvp))

y_breakpoints_mvp = []

for b in clean_breakpoints_mvp:
    values_at_breakpoint = []
    for fn in functions_mvp:
        values_at_breakpoint.append(fn(b*60))
        # print(f'{b} and {fn(b)}')
    y_breakpoints_mvp.append(np.mean(values_at_breakpoint))

plt.plot(clean_breakpoints_mvp,y_breakpoints_mvp, label = 'full MVP calculation')
#plt.scatter(clean_breakpoints_mvp, y_breakpoints_mvp)



########################
# Simple MVP Calculation
########################

average_breakpoints = [0,np.mean(first),np.mean(second),np.mean(third),np.mean(fourth)]#, np.mean(fifth), np.mean(sixth), np.mean(seventh), np.mean(eighth), np.mean(ninth), np.mean(tenth)]

# average_breakpoints = cost_jump_arrivals(lambda_parameter,round_T)

simple_f = create_piecewise_function(average_breakpoints, slope = m, surge = s)
y_simple = simple_f(x*60)

plt.plot(x,y_simple,label='simplified MVP calculation')

# Add a legend, labels, and show the plot
plt.xlabel("delay hours")
plt.ylabel("accumulated cost ($)")
plt.title("MVP Calculation")
plt.legend()
plt.show()


# import os 
# import pandas as pd
# path = os.getcwd()+'/test_stochastic/intra-alt-intra/2019-05-01/baseline/intra1/df_gurobi.csv'
# df_nonstoch = pd.read_csv(path)
# plt.scatter(df_nonstoch['new_delay_15bin_weighted'])
# print(df_nonstoch)