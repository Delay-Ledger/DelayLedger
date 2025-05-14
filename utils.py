# %%
# packages
import pandas as pd
pd.set_option('display.max_rows', 75)
import numpy as np
from datetime import datetime, timedelta, date
import json
import os
import matplotlib.pyplot as plt
import re
from scipy.stats import poisson

# gurobi
import gurobipy as gp
from gurobipy import GRB

from params import *


# def expected_pv(T, lambda_parameter, slope, surge):
#     # slope is in dollars per minute, T is in hours
#     # must get a dollars per minute expected value slope
#     return (T * 60 * slope + surge * (1/lambda_parameter) * T)/T/60

# def piecewise_constraints():
#     # new constraints
#     piecewise_breakpoints, functions_value_breakpoints = mean_value_objective_function(500, pv[f])
#     B = len(piecewise_breakpoints)

#     # add vars
#     tau_tuples = [(f, j) for f in F for j in range(B)]

#     y_tuples = [(f, j) for f in F for j in range(1, B)]

#     tau = model.addVars(tau_tuples, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="tau")  # j = 0..|B|
#     y = model.addVars(y_tuples, vtype=GRB.BINARY, name="y")  # j = 1..|B|

#     # add constraints
#     # i want these to be valid for all f in F
#     model.addConstr(tau[0] <= y[1], name="tau0_leq_y1")
#     for j in range(1, len(B)-1):
#         model.addConstr(tau[j] <= y[j] + y[j+1], name=f"tau_leq_{j}_{j+1}")
#     model.addConstr(tau[len(B)-1] <= y[len(B)-1], name="tau_last_leq")

#     # sum constraints
#     # i want these to be valid for all f in F
#     model.addConstr(gp.quicksum(tau[j] for j in B) == 1, name="sum_tau")
#     model.addConstr(gp.quicksum(y[j] for j in B[1:]) == 1, name="sum_y")

#     # di[f] constraint
#     model.addConstr(
#         di[f] == gp.quicksum(piecewise_breakpoints[j] * tau[j] for j in range(B)),
#         name=f"link_delay_{f}"
#     )

#     # cost variable
#     costs = {j: functions_value_breakpoints[j] for j in range(B)}

def piecewise_constraints(model, F, pv, di):
    """
    Adds flight-specific piecewise delay constraints for all flights f in F.

    Args:
        model: Gurobi model
        F: list of flight identifiers
        pv: dict mapping flight f to its private valuation
        di: delay variables in model

    Returns:
        tau: Gurobi tupledict for tau[f, j]
        y: Gurobi tupledict for y[f, j]
        costs_dict: {f: {j: cost}} for reference
        costs: (f,j) tuple dictionary
    """

    tau = {}  # (f, j) → variable
    y = {}
    costs = {}  # (f, j) → variable

    for f in F:
        # Flight-specific breakpoints and costs
        piecewise_breakpoints, functions_value_breakpoints = mean_value_objective_function(500, pv[f])
        piecewise_breakpoints.insert(0, -1*round_T*4) # maximum value in negative direction (bring flight earlier)
        functions_value_breakpoints.insert(0, -1*round_T*4*pv[f]) # value of cost decrease based on initial v_f

        B = len(piecewise_breakpoints)

        # cost function
        for j, val in enumerate(functions_value_breakpoints):
            costs[f, j] = val

        # Add tau[f,j] and y[f,j] variables
        for j in range(B):
            tau[f, j] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"tau_{f}_{j}")
        for j in range(1, B):
            y[f, j] = model.addVar(vtype=GRB.BINARY, name=f"y_{f}_{j}")

        # Constraint: τ_j ≤ y_j + y_{j+1}
        model.addConstr(tau[f, 0] <= y[f, 1], name=f"tau0_leq_y1_{f}")
        for j in range(1, B - 1):
            model.addConstr(tau[f, j] <= y[f, j] + y[f, j + 1], name=f"tau_leq_{f}_{j}_{j+1}")
        model.addConstr(tau[f, B - 1] <= y[f, B - 1], name=f"tau_last_leq_{f}")

        # Convex sum of τ_j and y_j
        model.addConstr(gp.quicksum(tau[f, j] for j in range(B)) == 1, name=f"sum_tau_{f}")
        model.addConstr(gp.quicksum(y[f, j] for j in range(1, B)) == 1, name=f"sum_y_{f}")

        # # Delay = sum(breakpoints[j] * τ_j)
        # model.addConstr(
        #     x[f] == gp.quicksum(piecewise_breakpoints[j] * tau[f, j] for j in range(B)),
        #     name=f"link_delay_{f}"
        # )

        # convex combination delay >= di_f
        model.addConstr(gp.quicksum(piecewise_breakpoints[j] * tau[f, j] for j in range(B)) >= di[f])

    from gurobipy import tupledict
    return tupledict(tau), tupledict(y), costs

def mean_value_objective_function(n,vf):
    """
    This function samples many iterations of the stochastically-generated
    piecewise linear functions and finds an average of all of the problems
    in an attempt to create the mean value problem.

    First, n number of samples are generated. Each of these samples will have
    their own set of breakpoints (at which surges in cost rate will occur). 
    Then, the mean function takes all of these breakpoints and calculates the
    average across all samples for each point.

    n: number of samples to draw
    vf: the starting flight value for a given flight

    Returns: two arrays
    mean_breakpoints: all breakpoints that were generated from sampling
    mean_y: average across all functions for each breakpoint

    note: the arrivals array will always have 0 in its first index
    """
    functions_list = []
    breakpoints_list = []
    for i in range(0,n):
        # sample a set of linear breakpoints (cost rate surges)
        breakpoints = cost_jump_arrivals(lambda_parameter,round_T)
        # record all the mean function
        for j in range(len(breakpoints)):
            breakpoints_list.append(breakpoints[j])

        # create a piecewise function from this sample
        f = create_piecewise_function(breakpoints, slope=vf, surge=surge)
        # record all functions for the mean function
        functions_list.append(f)
    
    breakpoints_list.append(round_T*4) # adding the end point of the round as well

    mean_breakpoints = sorted(set(breakpoints_list))
    # print("Number of Breakpoints:", len(clean_breakpoints_mvp))
    # print("Number of Function Samples:", len(functions_mvp))

    mean_y = []

    for b in mean_breakpoints:
        values_at_breakpoint = []
        for fn in functions_list:
            values_at_breakpoint.append(fn(b)) 
        mean_y.append(np.mean(values_at_breakpoint))

    return mean_breakpoints, mean_y

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
    while arrivals[i] < T*4:
        # randomly sample exponential variable with parameter lambda
        v = np.random.exponential(lambda_parameter)
        v_hours = np.floor(v)
        v_15mins = np.round(((v - v_hours)*60)/15) # fractional part of hour -> minutes -> 15min bins

        v_15bin = v_hours*4 + v_15mins
        
        # we are only interested in costs within the round
        if v_15bin + arrivals[i] > T*4:
            break

        # add cumulative values to arrivals array
        arrivals.append(v_15bin + arrivals[i])

        #iterate
        i += 1

    return arrivals

def create_piecewise_function(breakpoints, slope, surge):
    """
    NEW: need to change piecewise breakpoints output to be in 15min bins, then this function needs to output a function based on 15 min bins

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

    # Pre-first segment: constant slope before breakpoints[0]
    function_list.append(lambda x: slope * x) # for x < breakpoints[0] (=0)

    # Create function segments dynamically
    for i in range(num_segments):
        if i == 0:
            tracker_f = lambda x: slope * x
            function_list.append(lambda x: slope * x)  # First segment (before first breakpoint)
        else:
            # Apply cost surge at each breakpoint
            # Breakpoints are in hours, convert to minutes

            y_piecewise = tracker_f(breakpoints[i])  #breakpoints[i] * 60
            # plt.plot(breakpoints[i], 0, marker='o', linestyle='None', color='red', markersize = 2)  # 'o' = dot marker

            # y_piecewise = tracker_f(breakpoints[i] * 60) + surge

            # y = ax + b --> b =  y - ax
            adjusted_slope = slope + i * surge
            intercept = y_piecewise - adjusted_slope * breakpoints[i] # Solve for new function
            tracker_f = lambda x, a=adjusted_slope, b=intercept: a * x + b  # Update tracker

            # intercept = y_piecewise - slope * breakpoints[i] * 60
            # tracker_f = lambda x, intercept=intercept: slope * x + intercept #update tracker

            # Add next piecewise interval
            function_list.append(lambda x, a=adjusted_slope, b=intercept: a * x + b) 
            # function_list.append(lambda x, intercept=intercept: slope * x + intercept)

    # Create conditions as boolean masks to pass into np.piecewise function
    def condition_masks(x):
        masks = [x < breakpoints[0]]
        masks += [(breakpoints[i] <= x) & (x < breakpoints[i+1]) for i in range(num_segments - 1)]
        masks.append(x >= breakpoints[-1])  # Last condition (x ≥ last breakpoint)
        return masks

    return lambda x: np.piecewise(x, condition_masks(x), function_list)

# def create_piecewise_function(breakpoints, slope, surge):
#     """
#     Creates a piecewise function with dynamically generated function mappings.
#     The number of breakpoints are random, so a dynamic function was needed.
#     Each piecewise break defines the moment when a cost surge. The breaks
#     are in units of hours (since the round length is in hours), so we have
#     to be careful to change these to units of minutes, as this functions 
#     takes minutes.
#     The values of the slopes and surge for each piecewise interval may be 
#     different in the future, but they are set to the same value right now.

#     breakpoints: list of arrival times for each cost surge, in hours
#     slope: the slope of the linear segments, in units of dollars per minute
#     surge: the magnitude of the cost surge at each piecewise breakpoint

#     Returns: a function that evaluates piecewise expressions
#     """

#     num_segments = len(breakpoints)
    
#     # tracker_f created in order to track function values during piecewise construction
#     tracker_f = lambda x: slope * x  

#     # List to store each sub-function as a part of the piecewise function
#     function_list = []

#     # Create function segments dynamically
#     for i in range(num_segments):
#         if i == 0:
#             tracker_f = lambda x: slope * x
#             function_list.append(lambda x: slope * x)  # First segment (before first breakpoint)
#         else:
#             # Apply cost surge at each breakpoint
#             # Breakpoints are in hours, convert to minutes
#             y_piecewise = tracker_f(breakpoints[i] * 60) + surge  
#             # y = ax + b --> b =  y - ax
#             intercept = y_piecewise - slope * breakpoints[i] * 60 # Solve for new function
#             tracker_f = lambda x, intercept=intercept: slope * x + intercept  # Update tracker
#             # Add next piecewise interval
#             function_list.append(lambda x, intercept=intercept: slope * x + intercept) 

#     # Create conditions as boolean masks to pass into np.piecewise function
#     def condition_masks(x):
#         masks = [(breakpoints[i]*60 <= x) & (x < breakpoints[i+1]*60) for i in range(len(breakpoints) - 1)]
#         masks.append(x >= breakpoints[-1]*60)  # Last condition (x ≥ last breakpoint)
#         return masks

#     return lambda x: np.piecewise(x, condition_masks(x), function_list)
# %%
def record_changes(df_new, df_base, subdir_full_path, this_date, df_change_name = 'df_changes.csv'):
    # example inputs
    """
    # read new results
    df_new = pd.read_csv('d1/intra-alt-intra/2019-06-08/Control-UA/intra2/df_gurobi.csv')
    df_new.head()

    # read baseline
    df_base = pd.read_csv('d1/intra-alt-intra/2019-06-08/baseline/df_gurobi.csv')
    df_base.head()

    this_date = '2019-06-08'
    high_priority_b = 7
    low_priority_a = 3
    subdir_full_path = 'd1/intra-alt-intra'
    """
    
    # Counts
    base_series = df_new.groupby('marketing_airline_network')['new_delay_15bin'].count()

    # Read df_changes or create if doesn't exist
    if not os.path.exists(subdir_full_path+'/' + df_change_name):
        # initialize results df
        
        frame = {this_date+'-Count': base_series}
        df2 = pd.DataFrame(frame)
    else:
        df2 = pd.read_csv(subdir_full_path+'/' + df_change_name, index_col=0)
        df2[this_date+'-Count'] = base_series

    # add baseline results
    base_delay = dict(zip(df_base.flt_name, df_base.new_delay_15bin))
    df_new['baseline_delay_15bin'] = df_new['flt_name'].map(base_delay)  
    df_new['change_delay_15bin'] = df_new['new_delay_15bin'] - df_new['baseline_delay_15bin']

    # percent of flights moved earlier, later
    bool_earlier = df_new.new_delay_15bin < df_new.baseline_delay_15bin
    df2[this_date+'-num_earlier'] = np.round(df_new[bool_earlier].groupby('marketing_airline_network').size())
    df2[this_date+'-frac_earlier'] = np.round(df_new[bool_earlier].groupby('marketing_airline_network').size() / df2[this_date+'-Count'],3)
    df2[this_date+'-frac_earlier_high'] = np.round(df_new[bool_earlier & (df_new.flight_val >= high_priority_b)].groupby('marketing_airline_network').size() / df2[this_date+'-Count'],3)
    df2[this_date+'-frac_earlier_med'] = np.round(df_new[bool_earlier & (df_new.flight_val < high_priority_b) & (df_new.flight_val > low_priority_a)].groupby('marketing_airline_network').size() / df2[this_date+'-Count'],3)
    df2[this_date+'-frac_earlier_low'] = np.round(df_new[bool_earlier & (df_new.flight_val <= low_priority_a)].groupby('marketing_airline_network').size() / df2[this_date+'-Count'],3)


    bool_later = df_new.new_delay_15bin > df_new.baseline_delay_15bin
    df2[this_date+'-num_later'] = np.round(df_new[bool_later].groupby('marketing_airline_network').size())
    df2[this_date+'-frac_later'] = np.round(df_new[bool_later].groupby('marketing_airline_network').size() / df2[this_date+'-Count'],3)
    df2[this_date+'-frac_later_med'] = np.round(df_new[bool_later & (df_new.flight_val < high_priority_b) & (df_new.flight_val > low_priority_a)].groupby('marketing_airline_network').size() / df2[this_date+'-Count'],3)
    df2[this_date+'-frac_later_low'] = np.round(df_new[bool_later & (df_new.flight_val <= low_priority_a)].groupby('marketing_airline_network').size() / df2[this_date+'-Count'],3)

    # mean and stdev
    df_new['change_delay_15bin'] = df_new['change_delay_15bin'].astype('float')
    df2[this_date+'-earlier_mean'] = np.round(df_new[bool_earlier].groupby('marketing_airline_network')['change_delay_15bin'].mean(),2)
    df2[this_date+'-earlier_std'] = np.round(df_new[bool_earlier].groupby('marketing_airline_network')['change_delay_15bin'].std(),2)
    df2[this_date+'-later_mean'] = np.round(df_new[bool_later].groupby('marketing_airline_network')['change_delay_15bin'].mean(),2)
    df2[this_date+'-later_std'] = np.round(df_new[bool_later].groupby('marketing_airline_network')['change_delay_15bin'].std(),2)

    # save df2
    df2.fillna(0, inplace=True)
    df2.to_csv(subdir_full_path+'/' + df_change_name)


# %%

def create_df(input_csv, read_path, date_of_interest, untruthful_airline = ''):
    """
    Function cleans data and outputs df1 which contains rows of flights
    Inputs
    - input_csv: (string) filename
    - read_path: (string) directory where df1 will be saved
    - date_of_interest: (string) date to select
    - untruthful_airline: (string) of airline code that is untrutful and will have flight_val of untruthful_val
    Outputs
    - df1: (df) dataframe of day, saved as csv too
    - airline_counts: (dict) number of flights per airline
    """
    # read data
    df = pd.read_csv(input_csv)
    df = df.iloc[:,:-1]     # drop last column
    print(df.columns.tolist())
    print('Number of flights in month: {}'.format(df.shape))

    # read timezone data
    df_tz = pd.read_csv('airports_tz.csv', header=None)
    # df_tz = df_tz[(df_tz[3] == 'Canada') | (df_tz[3] == 'United States')]

    # convert timezones to ET
    # value is number of hours to add to local time to get ET
    tz_dict = dict(zip(df_tz[4],df_tz[9]))
    for key in tz_dict:
        try:
            tz_dict[key] = -5 - float(tz_dict[key])
        except:
            pass

    # Kearney, Nebraska
    tz_dict['EAR'] = 1

    # change column headers to lowercase
    df = df.rename(columns=str.lower)

    # includes previous day
    prev_date = (datetime.strptime(date_of_interest, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
    df_day = df[((df.flightdate == date_of_interest) | 
                ((df.flightdate == prev_date) & (df.deptime > df.arrtime))) &        # previous date
                ((df.origin.isin(ops30) | (df.dest.isin(ops30))))]                  # origin or destination is OPS30 airport

    # skips previous day
    df_day = df[(df.flightdate == date_of_interest) &                                         # previous date
                ((df.origin.isin(ops30) | (df.dest.isin(ops30))))]                  # origin or destination is OPS30 airport

    # omit small airlines like HA and G4
    df_day = df_day[~df_day['marketing_airline_network'].isin(['HA','G4'])]

    print('Number of flights on {}: {}'.format(date_of_interest, df_day.shape[0]))

    # fix 2400 times to 0
    df_day.loc[df_day.deptime > 2359, 'deptime'] = 0
    df_day.loc[df_day.arrtime > 2359, 'arrtime'] = 0

    # shift to ET
    df_day['orig_tz_adjust'] = df_day['origin'].map(tz_dict)*4      # 15min bins
    df_day['dest_tz_adjust'] = df_day['dest'].map(tz_dict)*4        # 15min bins

    # create columns for actual departure/arrival hour
    df_day['dep_hour'] = (df_day['deptime'] // 100)
    df_day['arr_hour'] = (df_day['arrtime'] // 100)

    df_day['dep_15min'] = np.round((df_day['deptime']-df_day['dep_hour']*100)/15)         # round, not //
    df_day['arr_15min'] = np.round((df_day['arrtime']-df_day['arr_hour']*100)/15)

    df_day['dep_15bin'] = df_day.dep_hour*4 + df_day.dep_15min
    df_day['arr_15bin'] = df_day.arr_hour*4 + df_day.arr_15min

    # create columns for departure/arrival hour
    df_day['crs_dep_hour'] = (df_day['crsdeptime'] // 100)
    df_day['crs_arr_hour'] = (df_day['crsarrtime'] // 100)

    df_day['crs_dep_15min'] = np.round((df_day['crsdeptime']-df_day['crs_dep_hour']*100) / 15)
    df_day['crs_arr_15min'] = np.round((df_day['crsarrtime']-df_day['crs_arr_hour']*100) / 15)

    df_day['crs_dep_15bin'] = df_day.crs_dep_hour*4 + df_day.crs_dep_15min
    df_day['crs_arr_15bin'] = df_day.crs_arr_hour*4 + df_day.crs_arr_15min

    # adjust timezones to ET
    for col in ['crs_dep_15bin','dep_15bin']:
        df_day[col] += df_day.orig_tz_adjust

    for col in ['crs_arr_15bin','arr_15bin']:
        df_day[col] += df_day.dest_tz_adjust

    # adjust overnight flights
    for col in ['crs_arr_15bin','arr_15bin']:
        df_day.loc[df_day.arr_15bin < df_day.dep_15bin, col] += 96         # 96 15-min bins in a day

    # filter by time
    df_day = df_day[df_day.crs_dep_15bin < crs_dep_thresh]
    print('flights remaining after filtering by time: {}'.format(df_day.shape[0]))

    # filter out cancelled and flights without tail numbers
    df1 = df_day[~(df_day.cancelled == 1) & ~(df_day.diverted == 1) & ~(df_day.tail_number.isna())]
    print('after filtering out cancelled flights and flights without tail, {} flights remain'.format(df1.shape[0]))

    # filter out excessively delayed flights
    df1 = df1[~(df1.nasdelay > 400)]
    print('after filtering out excessively nas delayed flights, {} flights remain'.format(df1.shape[0]))

    # save file
    if not os.path.exists(read_path):
        os.makedirs(read_path)


    # private flight value (between 1 and 10)
    np.random.seed(0)

    # NEW:
    # The valuation of flights (cost / min rate) are randomly generated
    # Then, a randomly generated piecewise function is created for each flight

    # Sample for flight cost
    # NEW: make capability if set to stochastic, then make flight_val just equal to slope
    # otherwise it should be expected vlaue
    # flight_val = np.random.randint(1, 10, df1.shape[0])

    # NEW: using the same random selections across all runs
    flight_val = np.loadtxt('eval_new_piecewise_30days/intra-alt-intra/'+date_of_interest+'/flight_val.csv', delimiter=",", skiprows=1, usecols=2, dtype=int)
    
    df1['flight_val'] = flight_val
    # flight_val.to_csv(date_of_interest+'_flightvals.csv', )
    pd.DataFrame(zip(df1['tail_number'], flight_val), columns=["tail_number", "flight_val"]).to_csv(read_path+'/flight_val.csv')

    # Expected flight value
    # df1['expected_flight_val'] = [expected_pv(T=round_T,lambda_parameter=lambda_parameter,slope=flight_val[i],
    #                                   surge=surge) for i in range(df1.shape[0])]
    # df1['expected_flight_val'] = flight_val # setting the objective function value to be the flight value as well

    df1['true_flight_val'] = df1['flight_val']
    # df1['true_expected_flight_val'] = df1['expected_flight_val']
    if untruthful_airline != '':
        df1.loc[(df1.marketing_airline_network == untruthful_airline), 'flight_val'] = untruthful_val

    # number of flights per airline
    airline_counts = df1.marketing_airline_network.value_counts().to_dict()

    # save df
    df1.to_csv(read_path+'/df1.csv', )

    return df1, airline_counts

def create_gurobi_vars(read_path, df1, untruthful_airline = '', input_dep_cap = {}, input_arr_cap = {}):
    """
    Create Gurobi variables necessary for optimization
    Inputs
    - read_path: (string) where to save
    - df1: (df) input dataframe, made by create_df function
    - bool_intra: (bool) True/False whether this is intra-airline deconfliction
    - input_capacity: contains capacities for use with intra-airline deconfliction, 
        particularly if alternating control already run
    Outputs
    - Updates and saves df1 to CSV
    """
    
    # set index
    df1 = df1.set_index(['tail_number','deptime']).sort_index(level=['tail_number','deptime'])

    """
    F_full: flight ID is "tail-airline-number of leg in day"
    """
    # number the flights in tail number
    df1['tail_flt_num'] = 0
    for tail in df1.index.get_level_values(0).unique():
        df1.loc[(tail),'tail_flt_num'] = np.arange(df1.loc[(tail),'tail_flt_num'].shape[0])
        df1.loc[(tail),'num_flts_tail'] = df1.loc[(tail),'tail_flt_num'].shape[0]-1

    df1['num_flts_after'] = df1.num_flts_tail-df1.tail_flt_num

    # create flight name
    df1['tail_number'] = df1.index.get_level_values(0)
    df1['flt_name'] = df1.tail_number + '-' + df1.marketing_airline_network + '-' + df1.tail_flt_num.astype(str)

    # update flight_val

    # # find the maximum number of flights after
    # max_flts_after = int(df1.num_flts_after.max())

    # # loop through this_flt_num and nfa (number of flights after this_flt_num)
    # for this_flt_num, nfa in enumerate(range(max_flts_after, 0, -1)):
    #     print('Processing flights that are {} in round, with {} flights after'.format(this_flt_num, nfa))
    #     # loop through rows with this_flt_num and nfa number of flights after
    #     for idx, row in df1[(df1.num_flts_after == nfa) & (df1.tail_flt_num == this_flt_num)].iterrows():
    #         start_str = row.tail_number+'-'+row.marketing_airline_network+'-'
    #         value_to_add = 0
    #         # loop through next flights after, starting with 1 after this flight
    #         # and going through the nfa number
    #         for j in range(1, nfa+1):
    #             flt_str = str(this_flt_num+j)
    #             value_to_add += df1.loc[df1.flt_name == start_str+flt_str,'flight_val'].iloc[0]
   
    #         df1.at[idx,'new_flight_val'] = row.flight_val + value_to_add  


    # even simpler
    if False:
        for idx, row in df1.iterrows():
            if row.num_flts_after <= 0:
                df1.at[idx,'new_flight_val'] = row.flight_val
                continue
            start_str = row.tail_number+'-'+row.marketing_airline_network+'-'
            this_flt_num = int(row.tail_flt_num)
            num_flts_after = int(row.num_flts_after)
            # columns to test are all possible next flights after (not all will exist)
            cols_to_add = [start_str+str(i) for i in range(this_flt_num+1, this_flt_num+num_flts_after+1)]
            value_to_add = sum([df1.loc[df1.flt_name == c, 'flight_val'].iloc[0] for c in cols_to_add])
            df1.at[idx,'new_flight_val'] = row.flight_val + value_to_add  

        # update flight_val (will need to fix true_flight_val later)
        # 2/14/22, don't do this, save this for another paper.
        # too complicated to define what the a and b are now. do airlines get to choose?
        # df1['flight_val'] = df1['new_flight_val']

    # # delete
    # df1['flight_val'] = df1['num_flts_after'] + 1



    # # simpler:
    # all_flt_names = df1.flt_name.values.tolist()
    # # loop through nfa (number of flights after), backward
    # for nfa in range(max_flts_after, 0, -1):
    #     # loop through rows with nfa number of flights after
    #     for idx, row in df1[(df1.num_flts_after == nfa)].iterrows():
    #         start_str = row.tail_number+'-'+row.marketing_airline_network+'-'
    #         this_flt_num = row.tail_flt_num
    #         # columns to test are all possible next flights after (not all will exist)
    #         cols_to_test = [start_str+str(i) for i in range(this_flt_num+1, this_flt_num+nfa+1)]
    #         if len(cols_to_test) > 0:
    #             value_to_add = [df1.loc[df1.flt_name == c, 'flight_val'].iloc[0] for c in cols_to_test if c in all_flt_names]
    #             df1.at[idx,'new_flight_val'] = row.flight_val + value_to_add  
                
    """
    T_dep: feasible departure times
    T_arr: feasible arrival times, based on T_dep and minimum travel time (tt)
    """
    print('96th percentile of nasdelay is: {} min'.format(df1.nasdelay.quantile(0.96)))

    # need to fix logic here
    # feasible departure time is between 
    # -- min: actual_dep - nasdelay - late ac delay
    df1['nasdelay'].fillna(0, inplace=True)
    df1['lateaircraftdelay'].fillna(0, inplace=True)
    df1['dep_earliest_15bin'] = df1.dep_15bin - np.round(df1.nasdelay/15) - np.round(df1.lateaircraftdelay/15)
    df1['dep_earliest_15bin'] = df1[['crs_dep_15bin','dep_earliest_15bin']].max(axis=1)

    # -- max: actual_dep + max_delay
    df1['dep_latest_15bin'] = df1.dep_15bin + max_delay_15bin

    # temporary fix
    df1.loc[df1.dep_latest_15bin < df1.dep_earliest_15bin, 'dep_latest_15bin'] += 96
    df1 = df1[df1.dep_earliest_15bin >= 0]      # ignore very early red-eyes
    df1 = df1[~(df1.tail_number == 'N211UA')]

    # feasible arrival time is between
    # -- min: min + actual_tt
    # -- max: min + actual_tt
    df1['actual_tt_15bin'] = df1.arr_15bin - df1.dep_15bin          
    df1['arr_earliest_15bin'] = df1['dep_earliest_15bin'] + df1['actual_tt_15bin']
    df1['arr_latest_15bin'] = df1['dep_latest_15bin'] + df1['actual_tt_15bin']

    # fix later
    # for now, filter arrival later than 200
    df1 = df1[df1.arr_latest_15bin < 130]
    print('after filtering out excessively late flights, {} remain'.format(df1.shape[0]))

    F = df1.flt_name.tolist()
    airline_ls = df1.marketing_airline_network.unique().tolist()
    # split F into airline list
    F_a = {}
    for a in airline_ls:
        F_a[a] = [x for x in F if '-'+a+'-' in x]

    print('first flight is {}'.format(F[0]))
    T_dep = {id:list(range(int(earliest),int(latest)+1)) for id, earliest, latest in zip(df1['flt_name'], df1['dep_earliest_15bin'], df1['dep_latest_15bin'])}
    T_arr = {id:list(range(int(earliest),int(latest)+1)) for id, earliest, latest in zip(df1['flt_name'], df1['arr_earliest_15bin'], df1['arr_latest_15bin'])}
    tt = {id:val for id, val in zip(df1['flt_name'], df1['actual_tt_15bin'])}

    print('feasible departure times of first flight is: {}'.format(T_dep[F[0]]))
    print('feasible arrival times of first flight is: {}'.format(T_arr[F[0]]))
    print('minimum travel time is: {}'.format(tt[F[0]]))

    T = list(range(0,int(df1.arr_latest_15bin.max())+1))
    print('\n last time period is : {}'.format(max(T)))

    """
    orig: origin
    dest: destination
    """
    # create origin and destination dictionaries
    orig = dict(zip(df1.flt_name,df1.origin))
    dest = dict(zip(df1.flt_name,df1.dest))

    print('first flight origin is: {}'.format(orig[F[0]]))
    print('first flight destination is: {}'.format(dest[F[0]]))

    """
    D: departure capacity, [airport k] [time t]
    A: arrival capacity, [airport k] [time t]
    k: airport list (could be just OPS30, or all airports)
    """
    D = {}
    A = {}
    min_capacity = 2

    # also, airline list
    D_a = {}
    A_a = {}

    if input_dep_cap == {}:
        for k in ops30:
            print('extracting capacity of', k)
            D[k] = {}
            A[k] = {}
            D_a[k] = {}
            A_a[k] = {}
            for t in T:
                # if t > 95:
                #     D[k][t] = 100
                #     A[k][t] = 100
                # else:
                D[k][t] = max(min_capacity,sum((df1.origin == k) & (df1.dep_15bin == t)))         # min value is 4
                A[k][t] = max(min_capacity,sum((df1.dest == k) & (df1.arr_15bin == t)))        # min value is 4

                # 2/2/22: Got rid of because it doesn't make sense to set airline capacities based on the data
                # Instead, run baseline MAGHP and then extract airline specific capacities
                # if bool_intra:
                #     D_a[k][t] = {}
                #     A_a[k][t] = {}
                #     for a in airline_ls:
                #         D_a[k][t][a] = sum((df1.origin == k) & (df1.dep_15bin == t) & (df1.marketing_airline_network == a)) 
                #         A_a[k][t][a] = sum((df1.dest == k) & (df1.arr_15bin == t) & (df1.marketing_airline_network == a)) 

    else:
        D_a = input_dep_cap
        A_a = input_arr_cap

    """
    scheduled times
    d: departure times
    r: arrival times
    """
    d = dict(zip(df1.flt_name, df1.dep_earliest_15bin))
    r = dict(zip(df1.flt_name, df1.arr_earliest_15bin))
    print('scheduled departure times of first flight is: {}'.format(d[F[0]]))
    print('scheduled arrival times of first flight is: {}'.format(r[F[0]]))

    df1['prev_arr_15bin'] = df1.groupby(level=0)['arr_15bin'].shift(1)
    df1['cxn_time'] = df1['dep_15bin'] - df1['prev_arr_15bin']
    df1.head().T

    # non-transferable delay
    df1['transferable_delay'] = np.round((df1.nasdelay + df1.lateaircraftdelay)/15)
    df1['transferable_delay'].fillna(0, inplace=True)

    """
    ct: minimum connection time between flight i and j operated by same tail
    cxns: flights i and j operated by same tail
    """

    # test
    # cxn_pairs = [('217NV-G4-0', '217NV-G4-1')]
    # ct = {}
    # ct[('217NV-G4-0', '217NV-G4-1')] = 2

    ct = {}

    # min_cxn_time = 4            # got rid of because too arbitrary

    # f precedes flight g
    for f in df1.flt_name.tolist():
        n = f.split('-')[-1]        # number of flight
        g = '-'.join(f.split('-')[:-1])+'-'+str(int(n)+1)
        if g not in df1.flt_name.tolist():
            continue
        else:
            # f_arr = df1[df1.flt_name==f]['arr_15bin'].values.tolist()[0]
            # g_dep = df1[df1.flt_name==g]['dep_15bin'].values.tolist()[0]

            # need to subtract transferable delay from connection time
            cxn_time = df1[df1.flt_name==g]['cxn_time'].values.tolist()[0]
            transferable_delay = df1[df1.flt_name==g]['transferable_delay'].values.tolist()[0]
            min_cxn_time = cxn_time - transferable_delay

            ct[(f,g)] = min_cxn_time

    cxn_pairs = list(ct.keys())

    # private flight value
    # NEW: expected_value in objective function
    pv = dict(zip(df1.flt_name, df1.flight_val))

    if untruthful_airline != '':
        pv = dict(zip(df1.flt_name, df1.true_flight_val))   
    

    # create high/low priority flight lists
    """
    F_high_priority: high priority flights dict
    F_med_priority: medium priority (but not low priority)
    F_low_priority: low priority
    """
    F_high_priority = {}
    F_med_priority = {}
    F_low_priority = {}

    airlines = df1.marketing_airline_network.drop_duplicates().values.tolist()

    # NEW: flight val and expected flight val together
    high_priority_b = 7
    low_priority_a = 3

    for airline in airlines:
        F_high_priority[airline] = df1[(df1.marketing_airline_network == airline) & (df1.flight_val >= high_priority_b)].flt_name.tolist()
        F_med_priority[airline] = df1[(df1.marketing_airline_network == airline) & (df1.flight_val < high_priority_b) & (df1.flight_val > low_priority_a)].flt_name.tolist()
        F_low_priority[airline] = df1[(df1.marketing_airline_network == airline) & (df1.flight_val <= low_priority_a)].flt_name.tolist()
    
        # check that all flights belong to some F_ list
        assert (len(F_high_priority[airline]) + len(F_med_priority[airline]) + len(F_low_priority[airline])) == df1[(df1.marketing_airline_network == airline)].shape[0]

    df1.to_csv(read_path+'/df1.csv', )

    return df1, F, F_a, T_dep, T_arr, tt, T, orig, dest, D, A, D_a, A_a, k, d, r, ct, cxn_pairs, pv, F_high_priority, F_med_priority, F_low_priority

def run_gurobi(save_path, ref_path, F, F_a, T_dep, T_arr, tt, T, orig, dest, D, A, D_a, A_a, k, d, r, ct, cxn_pairs, pv, F_high_priority, F_med_priority, F_low_priority, bool_intra,
    totalCapConstr=False, boolDelayCaps=False, airline_delay_caps = {},
    boolAirlineControl=False, airline_code='', max_increase_factor=1.3, boolPreIntra=False, boolFullSharing=False, boolFullSharingAbstract=False):
    """
    save_path: output path
    ref_path: path for baseline data
    F: list of flights
    F_a: dict of flights with key airline
    T_dep: dict of possible departure times
    T_arr: dict of possible arrival times
    tt: dict of travel times
    T: list of all time steps
    orig: dict of origins
    dest: dict of destinations
    D_a: dict of origins by airline
    A_a: dict of destinations by airline
    r: scheduled arrival times
    ct: minimum connection time between flight i and j operated by same tail
    cxns: flights i and j operated by same tail
    pv: private valuation
    """


    """
    INITIALIZE GUROBI VARIABLES
    """
    print('initializing gurobi variables...')
    model = gp.Model()

    # need to create list of tuples where entry is (flight, time)
    depart_var_tuples = []
    for f in F:
        for t in T_dep[f]:
            depart_var_tuples.append((f,t))

    arrive_var_tuples = []
    for f in F:
        for t in T_arr[f]:
            arrive_var_tuples.append((f,t))

    v = model.addVars(depart_var_tuples, vtype=GRB.BINARY, name='v')
    w = model.addVars(arrive_var_tuples, vtype=GRB.BINARY, name='w')
    di = model.addVars(F, vtype=GRB.INTEGER, name='di')
    # ddi = model.addVars(F, vtype=GRB.INTEGER, name='ddi')

    # high priority variables
    F_high_priority_ls = list(F_high_priority.values())
    F_high_priority_ls = [item for sublist in F_high_priority_ls for item in sublist]

    # medium priority variables
    F_med_priority_ls = list(F_med_priority.values())
    F_med_priority_ls = [item for sublist in F_med_priority_ls for item in sublist]
    mdi = model.addVars(F_med_priority_ls, vtype=GRB.INTEGER, name='mdi')

    F_low_priority_ls = list(F_low_priority.values())
    F_low_priority_ls = [item for sublist in F_low_priority_ls for item in sublist]
    ldi = model.addVars(F_low_priority_ls, vtype=GRB.INTEGER, name='ldi')   # low priority delay increase
    # mdd = model.addVars(F_med_priority_ls, vtype=GRB.INTEGER, name='mdd')

    """
    CONSTRAINTS
    """
    print('initializing constraints...')
    airline_ls = F_a.keys()

    if totalCapConstr:
        # (5): combined capacity
        model.addConstrs((sum(v[f,t] for f in F if orig[f] == k and t in T_dep[f]) +
                        sum(w[f,t] for f in F if dest[f] == k and t in T_arr[f]) <= D[k][t] + A[k][t]
                            for t in T
                            for k in ops30), name='dep_cap');
    else:
        if not bool_intra:
            # (5): departure capacity
            model.addConstrs((sum(v[f,t] for f in F if orig[f] == k and t in T_dep[f]) <= D[k][t]
                                for t in T
                                for k in ops30), name='dep_cap');

            # (6): arrival capacity
            model.addConstrs((sum(w[f,t] for f in F if dest[f] == k and t in T_arr[f]) <= A[k][t]
                                for t in T 
                                for k in ops30), name='arr_cap');

        else:
            # airline specific capacity (for airline_code)
            model.addConstrs((sum(v[f,t] for f in F_a[airline_code] if orig[f] == k and t in T_dep[f]) <= D_a[k][t][airline_code]
                            for t in T
                            for k in ops30), name='dep_cap_airline');

            # airline specific arrival capacity (for airline_code)
            model.addConstrs((sum(w[f,t] for f in F_a[airline_code] if dest[f] == k and t in T_arr[f]) <= A_a[k][t][airline_code]
                            for t in T 
                            for k in ops30), name='arr_cap_airline');

    # (7): depart at some point
    model.addConstrs((sum(v[f,t] for t in T_dep[f]) == 1
                        for f in F), name='must_dep');

    # (8): arrive at some point
    model.addConstrs((sum(w[f,t] for t in T_arr[f]) == 1
                        for f in F), name='must_arr');

    # (9): minimum travel time
    model.addConstrs(((sum(t*w[f,t] for t in T_arr[f]) - (sum(t*v[f,t] for t in T_dep[f]))) 
                        >= tt[f]
                        for f in F), name='min_tt');

    # (10): connections
    # filter connections if intra-airline swapping--airlines only know about their connections
    # 2/14/22. Writing only "if bool_intra" is Wrong. Controlling airlines needs to make schedule feasible. Changed to if bool_intra and not boolAirlineControl.
    if bool_intra and not boolAirlineControl:          
        cxn_pairs = [tup for tup in cxn_pairs if any('-'+airline_code+'-' in flt for flt in tup)]
    print('Number of connection pairs:', len(cxn_pairs))
    model.addConstrs(((sum(t*v[g,t] for t in T_dep[g]) - (sum(t*w[f,t] for t in T_arr[f]))) 
                        >= ct[(f,g)]
                        for (f,g) in cxn_pairs), name='min_cxns')

    # model.write('gurobipy.lp')

    # (11): delay caps

    # arrival delay increase variable
    for f in F:
        feas_arr = T_arr[f]
        model.addConstr(di[f] >= (sum(w[f,t]*(t-r[f]) for t in feas_arr)))
        model.addConstr(di[f] >= 0)

    # # departure delay increase variable
    # for f in F:
    #     feas_dep = T_dep[f]
    #     model.addConstr(ddi[f] >= (sum(v[f,t]*(t-d[f]) for t in feas_dep)))
    #     model.addConstr(ddi[f] >= 0)

    # for f in F:
    #         model.addConstr(di[f] >= 0)

    # model.addConstrs(di[f,t] >= (sum(w[f,t]*(t-r[f]) for t in T_arr[f])) 
    #                             for f in F)      
    # print(di)  
    # model.addConstrs(di[f,t] >= 0 for f in F for t in T_arr[f])
    
    # delay increase caps
    if boolDelayCaps:
        for airline in airline_delay_caps:
            if airline != airline_code:
                print('adding {} delay cap'.format(airline))
                model.addConstr(sum(di[f]        # arrival delay
                                for f in F if '-'+airline+'-' in f) <= max_increase_factor*airline_delay_caps[airline], name=airline+'_cap')

        # also optional total delay cap
        total_delay_cap = sum(airline_delay_caps.values())
        model.addConstr(sum(di[f]        # arrival delay
                        for f in F) <= max_tot_increase_factor*total_delay_cap, name=airline+'_cap')


    # (12): private delay cost constraint
    # NEW:
    high_priority_b = 7
    low_priority_a = 3

    if boolAirlineControl:
        if boolPreIntra:
            data = {}
            opt_delay = {}

            # loop through airlines
            airline_ls = F_a.keys()

            for airline in airline_ls:
                print('loading ', airline)
                with open(ref_path+'/baseline/Swap-'+airline+'/arrtimes.json') as f:
                    new_data = json.load(f)

                data = {**new_data, **data}

            # find baseline delay
            for f in F:
                arrival_time = int(min([float(i) for i in data[f] if round(data[f][i]) == 1.0]))
                opt_delay[f] = max(0, arrival_time - r[f])

        else:
            # read system optimal delay
            with open(ref_path+'/baseline/arrtimes.json') as f:
                data = json.load(f)

            opt_delay = {}

            for f in F:
                arrival_time = int(min([float(i) for i in data[f] if round(data[f][i]) == 1.0]))
                opt_delay[f] = max(0, arrival_time - r[f])

            # for f in F:
            #     possible_times = []
            #     for t in data[f]:
            #         if round(data[f][t]) == 1.0:
            #             possible_times.append(t)
            #     opt_delay[f] = max(0, int(min(possible_times)) - r[f])          # assigned arrival - scheduled arrival
            #     # opt_delay[f] = int(min([float(i) for i in data[f] if round(data[f][str(i)]) == 1.0]))


        for airline in airline_delay_caps:
            # high priority delay does not increase
            for f in F_high_priority[airline]:
                model.addConstr(di[f] <= opt_delay[f])

            # # medium priority delay variable
            # for f in F_med_priority[airline]:
            #     # delay increase
            #     model.addConstr(mdi[f] >= di[f] - opt_delay[f])
            #     model.addConstr(mdi[f] >= 0)

            #     # delay decrease
            #     model.addConstr(mdd[f] >= opt_delay[f] - di[f])
            #     model.addConstr(mdd[f] >= 0)       

            # low priority delay *increase*
            for f in F_low_priority[airline]:
                # delay increase
                model.addConstr(ldi[f] >= di[f] - opt_delay[f])
                model.addConstr(ldi[f] >= 0)             
            
              

            # medium priority delay leq opt_delay
            for f in F_med_priority[airline]:
                model.addConstr(di[f] <= opt_delay[f])

            # high priority delay savings outweigh low priority delay costs
            if airline != airline_code:
                print('adding {} private delay constraint'.format(airline))
                model.addConstr(high_priority_b * sum(opt_delay[f] - di[f]        # high priority delay savings
                                for f in F_high_priority[airline] if '-'+airline+'-' in f) 
                                
                                # + low_priority_a * sum(mdd[f]        # MIN medium priority delay savings
                                # for f in F_med_priority[airline] if '-'+airline+'-' in f)

                                # 2/10/22: this only works if low priority flights are constrained to only move back
                                # otherwise, low priority flights that move forward would have a benefit of low_priority_a,
                                # when, in reality, their benefit is [0, low_priority_a]. That is, we are overestimating benefit!
                                # >= low_priority_a * sum(di[f] - opt_delay[f]        # low priority delay costs
                                # for f in F_low_priority[airline] if '-'+airline+'-' in f))            

                                >= low_priority_a * sum(ldi[f]          # low priority delay costs
                                for f in F_low_priority[airline] if '-'+airline+'-' in f))            


                                # + high_priority_b * sum(mdi[f]        # MAX medium priority delay increase
                                # for f in F_med_priority[airline] if '-'+airline+'-' in f), name='pvd-'+airline)

    model.update()
    # model.write('gurobi.lp')

    # if boolDelayCaps:
    #     for airline in airline_delay_caps:
    #         if airline != airline_code:
    #             print('adding {} delay cap'.format(airline))
    #             model.addConstr(sum(sum(w[f,t]*(t-r[f]) for t in T_arr[f])        # arrival delay
    #                             for f in F if airline in f) <= max_increase_factor*airline_delay_caps[airline])

    """
    Piecewise CONSTRAINTS
    """
    print('initializing piecewise constraints...')

    F_all = list(F)

    # 05/06 changing bool logic to make changes before the functions, refer to standard file for original
    if boolAirlineControl or bool_intra:
        # if airline in control or running intra-deconfliction, only care about their own flights
        F = [f for f in F if airline_code in f]
    else:
        if boolFullSharing:
            if boolFullSharingAbstract:
                for f in F:
                    if f in F_high_priority_ls:
                        pv[f] = high_priority_b
                    if f in F_med_priority_ls:
                        pv[f] = low_priority_a
                    if f in F_low_priority_ls:
                        pv[f] = 0

    y = model.addVars(F,name="piecewise_cost")
    x = model.addVars(F,name="piecewise_delay")

    for f in F:
        piecewise_breakpoints, function_value_breakpoints = mean_value_objective_function(500, pv[f])

        # adding negative value capabilities
        piecewise_breakpoints.insert(0, -1*round_T*4) # maximum value in negative direction (bring flight earlier)
        function_value_breakpoints.insert(0, -1*round_T*4*pv[f]) # value of cost decrease based on initial v_f

        # x varaible needs to act like delay
        model.addConstr(x[f] >= di[f])
        # model.addConstr(x[f] >= 0) # the delay can be negative (bring flight earlier)
        model.addGenConstrPWL(x[f], y[f], piecewise_breakpoints, function_value_breakpoints, name=f"pwl_cost_function_{f}")
    
    # piecewise variables
    # tau, y, costs = piecewise_constraints(model, F, pv, di)

    model.update()

    """
    OBJECTIVE FUNCTION
    """
    print('initializing objective function...')
    print('optimizing delay only')

    c_a = 3
    c_g = 1

    if boolAirlineControl or bool_intra or boolFullSharing or boolFullSharingAbstract:
        model.setObjective(
            # + c_a*sum(pv[f]*sum(w[f,t]*(t-r[f]) for t in T_arr[f])
            #         - sum(v[f,t]*(t-d[f]) for t in T_dep[f])        # airborne delay for full flights only
            #             for f in F)
            #+ c_g*sum(pv[f][1]*di[f] #sum(w[f,t]*(t-r[f]) for t in T_arr[f])        # arrival delay
            #        for f in F)
            # + c_g*sum(pv[f]*di[f]
            #           for f in F)
            + c_g*sum(y[f] #sum  #arrival delay 
                        for f in F)
            # + c_g*sum(costs[f,j] * tau[f, j]
            #           for (f,j) in costs)
            # + c_g*sum(pv[f]*ddi[f] #sum(v[f,t]*(t-d[f]) for t in T_dep[f])        # ground delay
            #         for f in F)
            , GRB.MINIMIZE);
    else:
        model.setObjective(
                    + c_g*sum(di[f] #sum(w[f,t]*(t-r[f]) for t in T_arr[f])        # arrival delay
                            for f in F)
                    , GRB.MINIMIZE);

    model.update();

    # print('N7865A-WN-7' in [x[0] for x in depart_var_tuples])
    # print('N7865A-WN-7' in [x[0] for x in arrive_var_tuples])

    # print('N7865A-WN-7' in T_arr)
    # print('N7865A-WN-7' in T_dep)

    # print('N7865A-WN-7' in r)

    print('starting optimization...')
    model.Params.timeLimit = timelimit
    model.update();
    model.optimize()

    # Print number of solutions stored
    nSolutions = model.SolCount
    # print('Number of solutions found: ' + str(nSolutions))
    nSolutions = 1

    if model.status == 3:
        # do IIS
        print('The model is infeasible; computing IIS')
        model.computeIIS()
        if model.IISMinimal:
            print('IIS is minimal\n')
        else:
            print('IIS is not minimal\n')
        print('\nThe following constraint(s) cannot be satisfied:')
        for c in model.getConstrs():
            if c.IISConstr:
                print('%s' % c.constrName)
                
    # if boolAirlineControl:
    #     save_path += '/Control-'+airline_code
    # elif bool_intra:
    #     save_path += '/Swap-'+airline_code
    # else:
    #     save_path += '/baseline'

    for e in range(nSolutions):
        model.setParam(GRB.Param.SolutionNumber, e)
        print('\nSolution {}'.format(e+1))
        print('Objective Value: %g ' % model.ObjVal)

        # Output JSON file
        print('saving to ' + save_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        deptimes = dict(keys=F_all)
        arrtimes = dict(keys=F_all)
        

        # if model.status == GRB.OPTIMAL:
        #     for f in F_all:
        #         deptimes[f] = {}
        #         for t in T_dep[f]:
        #             deptimes[f][t] = v[f,t].x

        #     for f in F_all:
        #         arrtimes[f] = {}
        #         for t in T_arr[f]:
        #             arrtimes[f][t] = w[f,t].x
        # else:
        #     print("No solution available — model status:", model.status)

        # NEW: uncommented because this was the original

        for f in F_all:
            deptimes[f] = {}
            for t in T_dep[f]:
                deptimes[f][t] = v[f,t].x

        for f in F_all:
            arrtimes[f] = {}
            for t in T_arr[f]:
                arrtimes[f][t] = w[f,t].x

        # if boolAirlineControl:
        #     mdi_data = dict(keys=F_med_priority_ls)
        #     mdd_data = dict(keys=F_med_priority_ls)

        #     for f in F_med_priority_ls:
        #         mdi_data[f] = mdi[f].x
        #         mdd_data[f] = mdd[f].x

        #     with open(save_path+'/mdi_data.json', 'w') as outfile:
        #         json.dump(mdi_data, outfile, indent=4)

        #     with open(save_path+'/mdd_data.json', 'w') as outfile:
        #         json.dump(mdd_data, outfile, indent=4)


        with open(save_path+'/deptimes.json', 'w') as outfile:
            json.dump(deptimes, outfile, indent=4)

        with open(save_path+'/arrtimes.json', 'w') as outfile:
            json.dump(arrtimes, outfile, indent=4)

    return save_path

def create_df2(read_path, output_path, df1, F, untruthful_airline='', airline_ls=[]):
    """
    df1 is original dataframe, pre-Gurobi
    df2 is copy of df1 with Gurobi results from one run (baseline, airline control) or multiple (intra)
    read_path: where Gurobi results are saved, which will be read from
    output_path: where df_gurobi will be saved
    """
    
    df2 = df1.copy()
    df2['new_dep_15bin'] = -1
    df2['new_arr_15bin'] = -1

    # initial delay
    # initial_delay = df2[df2.init_delay_15bin > 0].init_delay_15bin.sum()
    # print('initial delay: {} 15-min bins'.format(initial_delay))       #[df1.init_delay_15bin > 0]

    # compute updated delay
    # just one file to read
    if len(airline_ls) == 0:
        with open(read_path+'/deptimes.json') as f:
            wo = json.load(f)

        with open(read_path+'/arrtimes.json') as f:
            wj = json.load(f)

        f_dep = {}
        f_arr = {}

        for f in F:
            f_dep[f] = int(min([float(i) for i in wo[f] if round(wo[f][i]) == 1.0]))
            f_arr[f] = int(min([float(i) for i in wj[f] if round(wj[f][i]) == 1.0]))

        # add actual times to df
        df2['new_dep_15bin'] = df2['flt_name'].map(f_dep)
        df2['new_arr_15bin'] = df2['flt_name'].map(f_arr)
    # multiple files to read
    else:
        f_dep = {}
        f_arr = {}
        for this_airline in airline_ls:
            # compute updated delay
            with open(read_path+'/Swap-'+this_airline+'/deptimes.json') as f:
                wo = json.load(f)
            with open(read_path+'/Swap-'+this_airline+'/arrtimes.json') as f:
                wj = json.load(f)

            # select flights from this_airline
            F_a = [x for x in F if '-'+this_airline+'-' in x]
            for f in F_a:
                f_dep[f] = int(min([float(i) for i in wo[f] if round(wo[f][i]) == 1.0]))
                f_arr[f] = int(min([float(i) for i in wj[f] if round(wj[f][i]) == 1.0]))

            # add actual times of this_airline to df
            df2['new_dep_15bin'] = df2['flt_name'].map(f_dep).fillna(df2['new_dep_15bin'])
            df2['new_arr_15bin'] = df2['flt_name'].map(f_arr).fillna(df2['new_arr_15bin'])

    # compute revised delay
    df2['new_delay_15bin'] = df2.new_arr_15bin - df2.arr_earliest_15bin
    # df2['change_delay_15bin'] = df2.new_delay_15bin - df2.init_delay_15bin

    # compute private value (even if airline control was not run)
    # NEW: piecewise calculation of functions
    # fix, we actually want the randomly generated slopes!
    
    # old
    # df2['new_delay_15bin_weighted'] = df2['new_delay_15bin'] * df2['expected_flight_val']

    # if untruthful_airline != '':
    #     df2['new_delay_15bin_weighted_true'] = df2['new_delay_15bin'] * df2['true_expected_flight_val']

    stochastic_delay_weighted = np.zeros((df2.shape[0],500))
    for x in range(500):
        delay_weighted = np.zeros(df2.shape[0])
        for idx in range(df2.shape[0]):
            # Generate piecewise breaks for each flight
            arrivals_array = cost_jump_arrivals(lambda_parameter, T=round_T) 
            f = create_piecewise_function(arrivals_array,df2['flight_val'].iloc[idx],surge)
            delay_weighted[idx] = f(df2['new_delay_15bin'].iloc[idx])
        stochastic_delay_weighted[:,x] = delay_weighted
    
    simulated_delay_weighted = np.mean(stochastic_delay_weighted, axis=1, keepdims=True)  # (x, 1) array

    df2['new_delay_15bin_weighted'] = simulated_delay_weighted

    # simulated_delay_weighted = np.zeros(df2.shape[0])
    # for idx in range(df2.shape[0]):
    #     # generate mean value function
    #     mvf_breakpoints, __ = mean_value_objective_function(500,df2['flight_val'].iloc[idx])
    #     # make a piecewise function to be able to evaluate decisions
    #     mvf_f = create_piecewise_function(mvf_breakpoints, df2['flight_val'].iloc[idx],surge)
    #     simulated_delay_weighted[idx] = mvf_f(df2['new_delay_15bin'].iloc[idx])

    # df2['new_delay_15bin_weighted'] = simulated_delay_weighted

    if untruthful_airline != '':
        stochastic_delay_weighted = np.zeros((df2.shape[0],500))
        for x in range(500):
            delay_weighted = np.zeros(df2.shape[0])
            for idx in range(df2.shape[0]):
                # Generate piecewise breaks for each flight
                arrivals_array = cost_jump_arrivals(lambda_parameter, T=round_T) 
                f = create_piecewise_function(arrivals_array,df2['true_flight_val'].iloc[idx],surge)
                delay_weighted[idx] = f(df2['new_delay_15bin'].iloc[idx])
            stochastic_delay_weighted[:,x] = delay_weighted
        
        simulated_delay_weighted = np.mean(stochastic_delay_weighted, axis=1, keepdims=True)  # (x, 1) array

        df2['new_delay_15bin_weighted_true'] = simulated_delay_weighted

        # simulated_delay_weighted = np.zeros(df2.shape[0])
        # for idx in range(df2.shape[0]):
        #     # generate mean value function
        #     mvf_breakpoints, __ = mean_value_objective_function(500,df2['true_flight_val'].iloc[idx])
        #     # make a piecewise function to be able to evaluate decisions
        #     mvf_f = create_piecewise_function(mvf_breakpoints, df2['true_flight_val'].iloc[idx],surge)
        #     simulated_delay_weighted[idx] = mvf_f(df2['new_delay_15bin'].iloc[idx])

        # df2['new_delay_15bin_weighted_true'] = simulated_delay_weighted
    
    # print('new delay: {} 15-min bins'.format(new_delay))        #[df2.new_delay_15bin > 0]
    # print('change in delay: {} 15-min bin'.format(df2.change_delay_15bin.sum()))
    # print('% reduction in delay: {} 15-min bins'.format((initial_delay-new_delay)/initial_delay))


    if not os.path.exists(output_path):
        os.mkdir(output_path)
    df2.to_csv(output_path+'/df_gurobi.csv', )

    
    return df2, f_dep, f_arr

def extract_airline_capacity(read_path, T, airline_ls, F, orig, dest):
    # Initialize dictionary
    D_a = {}
    A_a = {}

    for k in ops30:
        D_a[k] = {}
        A_a[k] = {}
        for t in T:
            D_a[k][t] = {}
            A_a[k][t] = {}
            for a in airline_ls:
                D_a[k][t][a] = 0
                A_a[k][t][a] = 0

    with open(read_path+'/deptimes.json') as f:
        wo = json.load(f)

    with open(read_path+'/arrtimes.json') as f:
        wj = json.load(f)

    # pattern = "-(.*?)-"     # used for identifying airline

    # loop through flights and add 1 to capacity for [airport][time][airline]
    for f in F:
        origin = orig[f]
        destination = dest[f]
        # airline = re.search(pattern, f).group(1)
        airline = f.split('-')[1]
        deptime = int(min([float(i) for i in wo[f] if round(wo[f][i]) == 1.0]))
        arrtime = int(min([float(i) for i in wj[f] if round(wj[f][i]) == 1.0]))

        if origin in D_a:
            D_a[origin][int(deptime)][airline] += 1
        if destination in A_a:
            A_a[destination][int(arrtime)][airline] += 1

    return D_a, A_a

def create_delay_series(df, untruthful_airline=''):

    # Get airline delay caps from baseline delay
    base_series = df.groupby('marketing_airline_network')['new_delay_15bin'].sum()
    base_series.sort_index(inplace=True)
    # airline_delay_caps = base_series.to_dict()
    # print(base_series)

    base_series_weighted = df.groupby('marketing_airline_network')['new_delay_15bin_weighted'].sum()
    base_series_weighted.sort_index(inplace=True)

    if untruthful_airline != '':
        base_series_weighted_true = df.groupby('marketing_airline_network')['new_delay_15bin_weighted_true'].sum()
        base_series_weighted_true.sort_index(inplace=True)
    else:
        base_series_weighted_true = ''

    return base_series, base_series_weighted, base_series_weighted_true

# %%
def color_boxplot(data, ax, pos, vert=True, color='k', widths=0.5):
    ax = ax or plt.gca()
    bp = ax.boxplot(data, patch_artist=True,  showmeans=False, positions=pos, whis=1000, widths=widths, vert=vert)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[item], color=color)
    for patch in bp['boxes']:
        patch.set(facecolor='w',linewidth=2)   
    for item in ['caps','fliers']:
        plt.setp(bp[item],linewidth=2)
    for item in ['whiskers']:
        plt.setp(bp[item],linewidth=2)
    for item in ['medians']:
        plt.setp(bp[item],linewidth=2)
    