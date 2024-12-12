# %%
# packages
import this
import pandas as pd
pd.set_option('display.max_rows', 75)
import numpy as np
import sys
from datetime import datetime, timedelta, date
import json
import os
import matplotlib.pyplot as plt

# gurobi
import gurobipy as gp
from gurobipy import GRB

# params
from params import *
from utils import record_changes, create_delay_series, create_df, create_gurobi_vars, run_gurobi, create_df2, extract_airline_capacity


# %%


# %%
def run_day(subdir_name, this_date, run_type, airline='', untruthful_airline='', max_increase_factor=1.3):
    """
    subdir_name: name for output folder
    this_date: date to run in "%Y-%m-%d" string
    run_type: 
    - 'intra-alt-intra' (Intra-airline swapping then Deled then Intra swapping again)
    - 'alt-intra'
    - 'full-sharing': airlines share private valuations 
    either exact values (boolFullSharing) or boolFullSharingAbstract (priority levels)
    - most commonly 'intra-alt-intra'
    airline: name of coordinating airline (string); leave empty if no coordinator
    max_increase_factor: how much delay can increase for individual airline relative to baseline MAGHP
    """

    # Run type
    print('RUN TYPE: {}'.format(run_type))
    bool_intra = False
    boolFullSharingAbstract = False

    if run_type == 'intra-alt-intra':
        boolPreIntra = True
    else:
        boolPreIntra = False

    if 'full-sharing' in run_type:
        boolFullSharing = True
        if 'abstract' in run_type:
            boolFullSharingAbstract = True
    else:
        boolFullSharing = False
        

    # Create directory and dataframe for the day (if necessary)
    subdir_full_path = os.getcwd()+'/'+subdir_name
    date_path = subdir_full_path+'/'+this_date
    if not os.path.exists(date_path+'/df1.csv'):
        df1, airline_counts = create_df(input_csv, date_path, this_date, untruthful_airline=untruthful_airline)
    else:
        print('df1 already exists, reading it')
        df1 = pd.read_csv(date_path+'/df1.csv')
        airline_counts = df1.marketing_airline_network.value_counts().to_dict()


    ### BASELINE
    # Create gurobi variables for Baseline
    df1, F, F_a, T_dep, T_arr, tt, T, orig, dest, D, A, D_a, A_a, k, d, r, ct, cxn_pairs, pv, F_high_priority, F_med_priority, F_low_priority = \
            create_gurobi_vars(date_path, df1, untruthful_airline=untruthful_airline)
    
    # cxn1 = ct[('N850AE-AA-0', 'N850AE-AA-1')]

    # tt1 =  tt['N850AE-AA-0']
    # tt2 = tt['N850AE-AA-1']

    # # Collect baseline departure and arrival times
    # baseline_d = d
    # baseline_r = r

    # Baseline: Run Gurobi
    airline_ls = []
    base_path = date_path + '/baseline'

    if os.path.exists(date_path+'/baseline/arr_times.json'):
        print('gurobi already run')
    else:
        print('Running baseline')
        run_gurobi(base_path, date_path, F, F_a, T_dep, T_arr, tt, T, orig, dest, D, A, D_a, A_a, k, d, r, ct, cxn_pairs, pv, F_high_priority, F_med_priority, F_low_priority, bool_intra,
            totalCapConstr=False, boolDelayCaps=False, airline_delay_caps = {},
            boolAirlineControl=False, airline_code='', max_increase_factor=max_increase_factor, 
            boolFullSharing = boolFullSharing, boolFullSharingAbstract=boolFullSharingAbstract)

    # Baseline: Save Gurobi Results
    df_base, _, _ = create_df2(base_path, base_path, df1, F, untruthful_airline=untruthful_airline, airline_ls=airline_ls)

    base_series, base_series_weighted, base_series_weighted_true = create_delay_series(df_base, untruthful_airline=untruthful_airline)
    airline_delay_caps = base_series.to_dict()

    # Create/update df_delays
    if not os.path.exists(subdir_full_path+'/df_delays.csv'):
        # enter this loop if first day
        frame = {this_date+'-Baseline': base_series}
        df_delays = pd.DataFrame(frame)
        airline = max(airline_delay_caps, key=airline_delay_caps.get)
        # delete later
        # airline = 'DL'
        print('After Baseline of first day, Airline in Control is:', airline)

        # # average delay
        # df_delays[this_date+'-Counts'] = df_delays.index.map(airline_counts)
        # df_delays[this_date+'-Baseline-avg'] = df_delays[this_date+'-Baseline']/df_delays[this_date+'-Counts']
        # df_delays[this_date+'-Baseline-Weighted'] = base_series_weighted

        if untruthful_airline != '':
            df_delays[this_date+'-Baseline-Weighted-True'] = base_series_weighted_true

    else:
        df_delays = pd.read_csv(subdir_full_path+'/df_delays.csv', index_col=0)
        df_delays[this_date+'-Baseline'] = base_series

    # Average Delay
    df_delays[this_date+'-Counts'] = df_delays.index.map(airline_counts)
    df_delays[this_date+'-Baseline-avg'] = df_delays[this_date+'-Baseline']/df_delays[this_date+'-Counts']

    # Baseline Weighted Delay
    df_delays[this_date+'-Baseline-Weighted'] = base_series_weighted
    if untruthful_airline != '':
        df_delays[this_date+'-Baseline-Weighted-True'] = base_series_weighted_true

    # Save df_delays
    df_delays.to_csv(subdir_full_path+'/df_delays.csv', )

    # Intra: Run Gurobi
    if run_type in ['just-intra', 'intra-alt-intra']:
        bool_intra = True      # SWITCH to True
        intra_path1 = base_path + '/intra1'             # where df_gurobi will be saved
        
        # Read updated capacities
        airline_ls = F_a.keys()
        D_a, A_a = extract_airline_capacity(base_path, T, airline_ls, F, orig, dest)
        
        for this_airline in airline_ls:
            print('Running intra-deconfliction for {}'.format(this_airline))
            this_airline_path = base_path + '/Swap-' + this_airline
            F_airline = F_a[this_airline]   # only pass in this airline's flights
            run_gurobi(this_airline_path, date_path, F_airline, F_a, T_dep, T_arr, tt, T, orig, dest, D, A, D_a, A_a, k, d, r, ct, cxn_pairs, pv, F_high_priority, F_med_priority, F_low_priority, bool_intra,
                totalCapConstr=False, boolDelayCaps=False, airline_delay_caps = {},
                boolAirlineControl=False, airline_code=this_airline, max_increase_factor=max_increase_factor)

        # Intra: Save Gurobi Results
        df_intra, f_dep, f_arr = create_df2(base_path, intra_path1, df1, F, untruthful_airline=untruthful_airline, airline_ls=airline_ls)

        base_series, base_series_weighted, base_series_weighted_true = create_delay_series(df_intra, untruthful_airline=untruthful_airline)
        airline_delay_caps = base_series.to_dict()

        # Update df_delays
        df_delays[this_date+'-Intra1'] = base_series
        df_delays[this_date+'-Intra1-avg'] = df_delays[this_date+'-Intra1']/df_delays[this_date+'-Counts']

        # Weighted Delay
        df_delays[this_date+'-Intra1-Weighted'] = base_series_weighted
        if untruthful_airline != '':
            df_delays[this_date+'-Intra1-Weighted-True'] = base_series_weighted_true

        # Save df_delays
        df_delays.to_csv(subdir_full_path+'/df_delays.csv', )


        # if '-alt-' in run_type:
        #     # Update desired departure and arrival times of flights
        #     d = f_dep
        #     r = f_arr

        #     # For controlling airline, use baseline departure and arrival times,
        #     # For participants, use updated departure and arrival times (post-intra)
        #     for f in d:
        #         if '-'+airline+'-' in f:
        #             d[f] = baseline_d[f]
        #             r[f] = baseline_r[f]
    
    # end if not running DLM
    if 'alt-' not in run_type:
        if 'intra' in run_type:
            # Save changes
            record_changes(df_intra, df_base, subdir_full_path, this_date)

        return airline
    

    ### AIRLINE-CONTROL

    # Run Gurobi (Alternating Control)
    bool_intra = False
    airline_path = date_path + '/Control-' + airline

    if os.path.exists(date_path+'/'+airline+'/arr_times.json'):
        print('gurobi already run')
        
    else:
        run_gurobi(airline_path, date_path, F, F_a, T_dep, T_arr, tt, T, orig, dest, D, A, D_a, A_a, k, d, r, ct, cxn_pairs, pv, F_high_priority, F_med_priority, F_low_priority, bool_intra,
            totalCapConstr=False, boolDelayCaps=True, airline_delay_caps = airline_delay_caps,
            boolAirlineControl=True, airline_code=airline, max_increase_factor=max_increase_factor, boolPreIntra=boolPreIntra)

    # Alt: Save Gurobi Results
    airline_ls = []         # only pull one file
    df_alt, _, _ = create_df2(airline_path, airline_path, df1, F, untruthful_airline=untruthful_airline, airline_ls=airline_ls)

    base_series, base_series_weighted, base_series_weighted_true = create_delay_series(df_alt, untruthful_airline=untruthful_airline)

    # Update df_delays
    df_delays[this_date+'-'+airline] = base_series

    # Weighted Delay
    df_delays[this_date+'-'+airline+'-Weighted'] = base_series_weighted
    if untruthful_airline != '':
        df_delays[this_date+'-'+airline+'-Weighted-True'] = base_series_weighted_true

    # Save average delay and increase relative to baseline (or Intra1??)
    df_delays[this_date+'-'+airline+'-avg'] = df_delays[this_date+'-'+airline]/df_delays[this_date+'-Counts']
    if run_type == 'alt-intra': #or run_type == 'intra-alt-intra':
        df_delays[this_date+'_incr-avg'] = df_delays[this_date+'-'+airline+'-avg'] - df_delays[this_date+'-Baseline-avg']
    else:
        df_delays[this_date+'_incr-avg'] = df_delays[this_date+'-'+airline+'-avg'] - df_delays[this_date+'-Intra1-avg']


    # Update ledger and pick next controlling airline

    # get column names of 5 previous dates
    # this_datetime = datetime.strptime(this_date, "%Y-%m-%d")
    # delta = timedelta(days=1)
    # date_ls = []
    # for _ in range(5):
    #     date_ls.append(this_datetime.strftime("%Y-%m-%d"))
    #     this_datetime = this_datetime-delta
        

    incr_cols = [c for c in df_delays.columns if '_incr-avg' in c] # and c[:10] in date_ls]   # this is rolling ledger
    df_delays[this_date+'_delay_ledger'] = df_delays[incr_cols].sum(axis=1)
    next_airline = df_delays[this_date+'_delay_ledger'].idxmax()
    # if next_airline == airline:
    #     print('Avoiding repeat...')
    #     next_airline = df_delays[this_date+'_delay_ledger'].nlargest(2).idxmin()
    print('After {}, Airline in Control will be: {}'.format(this_date,next_airline))

    # Save df_delays
    df_delays.to_csv(subdir_full_path+'/df_delays.csv', )

    ### Intra-Airline Swapping #2
    if True:        # post_alternating_swapping
        bool_intra = True           # SWITCH bool_intra to True, because now swapping
        airline_ls = F_a.keys()
        intra_path2 = airline_path + '/intra2'

        # read updated capacities
        D_a, A_a = extract_airline_capacity(airline_path, T, airline_ls, F, orig, dest)

        # # use baseline departure and arrival scheduled times to calculate delay
        # d = baseline_d
        # r = baseline_r

        # Run Gurobi (Intra-Airline Swapping)
        for this_airline in airline_ls:
            print('Running intra-deconfliction for {}'.format(this_airline))
            this_airline_path = airline_path + '/Swap-' + this_airline
            F_airline = F_a[this_airline]   # only pass in this airline's flights
            run_gurobi(this_airline_path, date_path, F_airline, F_a, T_dep, T_arr, tt, T, orig, dest, D, A, D_a, A_a, k, d, r, ct, cxn_pairs, pv, F_high_priority, F_med_priority, F_low_priority, bool_intra,
                totalCapConstr=False, boolDelayCaps=False, airline_delay_caps = {},
                boolAirlineControl=False, airline_code=this_airline, max_increase_factor=max_increase_factor)


    # Save Gurobi results
    df_airline, _, _ = create_df2(airline_path, intra_path2, df1, F, untruthful_airline=untruthful_airline, airline_ls=airline_ls)

    base_series, base_series_weighted, base_series_weighted_true = create_delay_series(df_airline, untruthful_airline=untruthful_airline)

    # Save Gurobi results
    df_delays[this_date+'-'+airline+'-Intra2'] = base_series
    df_delays[this_date+'-'+airline+'-Intra2-Weighted'] = base_series_weighted
    if untruthful_airline != '':
        df_delays[this_date+'-'+airline+'-Intra2-Weighted-True'] = base_series_weighted_true

    # Save df_delays
    df_delays.to_csv(subdir_full_path+'/df_delays.csv', )

    # Save changes
    if run_type == 'alt-intra':
        record_changes(df_airline, df_base, subdir_full_path, this_date)
    else:
        record_changes(df_airline, df_intra, subdir_full_path, this_date)


    # ### Track metrics
    # # Create/update df_metrics

    # # M1: count
    # base_series = df_airline.groupby('marketing_airline_network')['new_delay_15bin'].count()
    # base_series.sort_index(inplace=True)

    # if not os.path.exists(subdir_full_path+'/df_metrics.csv'):
    #     # enter this loop if first day
    #     frame = {this_date+'-Count': base_series}
    #     df_metrics = pd.DataFrame(frame)

    # else:
    #     df_metrics = pd.read_csv(subdir_full_path+'/df_metrics.csv', index_col=0)


    # # M2: offers
    # df_metrics[this_date+'-high_priority_set'] = df_airline[(df_airline.flight_val >= 7)].groupby('marketing_airline_network').size()
    # df_metrics[this_date+'-med_priority_set'] = df_airline[(df_airline.flight_val > 3) & (df_airline.flight_val < 7)].groupby('marketing_airline_network').size()
    # df_metrics[this_date+'-low_priority_set'] = df_airline[(df_airline.flight_val <= 3)].groupby('marketing_airline_network').size()

    # # M3: movement
    # df_metrics[this_date+'-high_priority_earlier'] = df_airline[(df_airline.flight_val >= 7) & (df_airline.change_delay_15bin<0)].groupby('marketing_airline_network').size()
    # df_metrics[this_date+'-med_priority_earlier'] = df_airline[(df_airline.flight_val > 3) & (df_airline.flight_val < 7) & (df_airline.change_delay_15bin<0)].groupby('marketing_airline_network').size()
    # df_metrics[this_date+'-low_priority_later'] = df_airline[(df_airline.flight_val <= 3) & (df_airline.change_delay_15bin>0)].groupby('marketing_airline_network').size()

    # # M4: delay
    # df_metrics[this_date+'-delay_reduced'] = df_airline[(df_airline.change_delay_15bin<0)].groupby('marketing_airline_network')['change_delay_15bin'].sum()*15
    # df_metrics[this_date+'-delay_increased'] = df_airline[(df_airline.change_delay_15bin>0)].groupby('marketing_airline_network')['change_delay_15bin'].sum()*15
    # df_metrics[this_date+'-delay_change'] = df_airline.groupby('marketing_airline_network')['change_delay_15bin'].sum()*15
    # df_metrics[this_date+'-delay_reduced_max'] = df_airline[(df_airline.change_delay_15bin<0)].groupby('marketing_airline_network')['change_delay_15bin'].min()*15
    # df_metrics[this_date+'-delay_increased_max'] = df_airline[(df_airline.change_delay_15bin>0)].groupby('marketing_airline_network')['change_delay_15bin'].max()*15

    # df_metrics.to_csv(subdir_full_path+'/df_metrics.csv', )

    return next_airline



# %%
# subdir_name = 'test1'
# this_date = '2019-06-21'
# run_day(subdir_name, this_date)


next_airline = ''                   # initially empty string
untruthful_airline = ''             # usually empty string

run_type = 'full-sharing'
run_type = 'intra-alt-intra' 

# sensitivity
max_increase_factor = 1.3


folder_name = '/home/yolandz/LATTICE/Deled/test_may'
folder_name = 'test_stepfunc'
#folder_name = 'test_normal_7_2'

save_name = folder_name + '/' + run_type
if not os.path.exists(save_name):
    os.makedirs(save_name)

# this_date = date(2019, 7,1)
# end_date = date(2019, 7, 31)

start_date_str = sys.argv[1]
end_date_str = sys.argv[2]
month = sys.argv[3]
input_csv = f'/home/yolandz/LATTICE/Deled/2019_ontime/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2019_{month}/On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2019_{month}.csv'
input_csv = 'On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2019_5.csv'
try:
    this_date = date.fromisoformat(start_date_str)
    end_date = date.fromisoformat(end_date_str)
except ValueError:
    print("Error: Please provide dates in the correct format (YYYY-MM-DD).")
    sys.exit(1)

delta = timedelta(days=1)
while this_date <= end_date:
    print(this_date.strftime("%Y-%m-%d"))
    # try:
    next_airline = run_day(save_name, this_date.strftime("%Y-%m-%d"), run_type, 
        airline=next_airline, untruthful_airline=untruthful_airline, max_increase_factor=max_increase_factor)
    # except:
    #     pass
    this_date += delta

# %%
