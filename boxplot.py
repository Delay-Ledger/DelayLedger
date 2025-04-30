import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, date
from utils import color_boxplot, create_piecewise_function, cost_jump_arrivals
from params import *

# boxplot
run_type = 'intra-alt-intra' 
folder_name = 'eval_lambda2_surge10'

subdir_name = folder_name + '/' + run_type

subdir_full_path = os.getcwd()+'/'+subdir_name

df_delays = pd.read_csv(subdir_full_path+'/df_delays.csv')

this_date = date(2019, 5, 1)
end_date = date(2019, 5, 4)
delta = timedelta(days=1)

coordinating_airline = pd.read_csv(subdir_full_path+'/coordinating_airline.csv',header=None)[0]
i=0
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))
delay_change_matrix = pd.DataFrame()
max_change = []
min_change = []
variance = []

while this_date <= end_date:
    coord_data_baseline = df_delays[df_delays["marketing_airline_network"] == coordinating_airline[i]][this_date.strftime("%Y-%m-%d")+'-Baseline-Weighted'].dropna()
    participant_data_baseline = df_delays[df_delays["marketing_airline_network"] != coordinating_airline[i]][this_date.strftime("%Y-%m-%d")+'-Baseline-Weighted'].dropna()

    coord_data_deled = df_delays[df_delays["marketing_airline_network"] == coordinating_airline[i]][this_date.strftime("%Y-%m-%d")+'-'+coordinating_airline[i]+'-Intra2-Weighted'].dropna()
    participant_data_deled = df_delays[df_delays["marketing_airline_network"] != coordinating_airline[i]][this_date.strftime("%Y-%m-%d")+'-'+coordinating_airline[i]+'-Intra2-Weighted'].dropna()

    coord_delay_change = coord_data_deled - coord_data_baseline
    participant_delay_change = participant_data_deled - participant_data_baseline
    
    # print(participant_delay_change)
    # print(df_delays[df_delays["marketing_airline_network"] != coordinating_airline[i]][["marketing_airline_network",this_date.strftime("%Y-%m-%d")+'-Baseline-Weighted']])

    color_boxplot(data=participant_delay_change, ax=ax1, pos=[i+1])
    ax1.scatter([i+1] * len(coord_delay_change), coord_delay_change, color=airline_color_dict[coordinating_airline[i]], label = coordinating_airline[i], marker='o', zorder=5)

    coord_percentage_change = (coord_data_deled - coord_data_baseline) / (coord_data_baseline) * 100
    participant_percentage_change = (participant_data_deled - participant_data_baseline) / (participant_data_baseline) * 100

    max_change.append(participant_percentage_change.max())
    min_change.append(participant_percentage_change.min())
    variance.append(participant_percentage_change.var())

    color_boxplot(data=participant_percentage_change, ax=ax2, pos=[i+1])
    ax2.scatter([i+1] * len(coord_percentage_change), coord_percentage_change, color = airline_color_dict[coordinating_airline[i]], label=coordinating_airline[i], marker='o', zorder=5)

    print("date:",this_date)

    # # only works with debug_test_stochastic because i recorded df_base only for this one
    # # Read Data
    # df_base = pd.read_csv(subdir_full_path+'/'+str(this_date)+'/df_base.csv')
    # df_alt = pd.read_csv(subdir_full_path+'/'+str(this_date)+'/df_alt.csv')

    # # Merge DataFrames on 'id'
    # merged_df = df_base.merge(df_alt, on=["flt_name","marketing_airline_network"],suffixes=("_base","_deled"))[["flt_name","marketing_airline_network","new_delay_15bin_base", "new_delay_15bin_deled", "new_delay_15bin_weighted_base","new_delay_15bin_weighted_deled","true_flight_val_deled","flight_val_deled","expected_flight_val_deled"]]

    # # Compute the difference
    # merged_df["difference"] = merged_df["new_delay_15bin_weighted_deled"] - merged_df["new_delay_15bin_weighted_base"]
    # merged_df["difference_non_w"] = merged_df["new_delay_15bin_deled"] - merged_df["new_delay_15bin_base"]

    # merged_df = merged_df.sort_values(by="difference", ascending=False)

    # merged_df.to_csv(subdir_full_path+'/'+str(this_date)+'/merged_df.csv')

    # print(merged_df[merged_df['difference'] > 0][['flight_val_deled','flt_name']].groupby("flight_val_deled").count())
    # print(merged_df[['marketing_airline_network','difference']].groupby("marketing_airline_network").sum())

    # # airline delay change
    # data_baseline = df_delays[this_date.strftime("%Y-%m-%d")+'-Baseline-Weighted'].dropna()
    # data_deled = df_delays[this_date.strftime("%Y-%m-%d")+'-'+coordinating_airline[i]+'-Intra2-Weighted'].dropna()

    # # IN PROGRESS: work to get a summary value to compare the deled with expected value and with normal function

    # if i == 0:
    #     delay_change_matrix['Airline'] = df_delays["marketing_airline_network"]

    # difference = (data_deled-data_baseline)/(data_baseline)*100
    # delay_change_matrix['Difference-'+this_date.strftime("%Y-%m-%d")] = difference

    this_date += delta
    i += 1

    # print("real")
    # print("low priority flights: ",merged_df[(merged_df['true_flight_val_deled'] <= low_priority_a)][["marketing_airline_network","difference"]].groupby("marketing_airline_network").sum())
    # print("medium priority flights: ",merged_df[(merged_df['true_flight_val_deled'] > low_priority_a) & (merged_df['flight_val_deled'] < high_priority_b)][["marketing_airline_network",'difference']].groupby("marketing_airline_network").sum())
    # print("high priority flights: ",merged_df[(merged_df['true_flight_val_deled'] >= high_priority_b)][["marketing_airline_network",'difference']].groupby("marketing_airline_network").sum())

    # print("constraint 5")
    # print("low priority flights: ",merged_df[(merged_df['flight_val_deled'] <= low_priority_a)][['marketing_airline_network','difference_non_w']].groupby("marketing_airline_network").sum()*3)
    # print("medium priority flights: ",merged_df[(merged_df['flight_val_deled'] > low_priority_a) & (merged_df['flight_val_deled'] < high_priority_b)][['marketing_airline_network','difference_non_w']].groupby("marketing_airline_network").sum()*3)
    # print("high priority flights: ",merged_df[(merged_df['flight_val_deled'] >= high_priority_b)][['marketing_airline_network','difference_non_w']].groupby('marketing_airline_network').sum()*7)


delay_change_matrix["Average Private Cost Change"] = delay_change_matrix.select_dtypes(include="number").mean(axis=1)

print("average max change",np.mean(max_change))
print("average min change",np.mean(min_change))
print("average variance",np.mean(variance))

# Compute the mean of numeric columns
average_row = delay_change_matrix.select_dtypes(include="number").mean()

# Convert it to a DataFrame and add a label
average_row["Airline"] = "Total"
delay_change_matrix = pd.concat([delay_change_matrix, pd.DataFrame([average_row])], ignore_index=True)

print(delay_change_matrix[['Airline','Average Private Cost Change']])

# Set title, labels, and format x-ticks
ax1.set_xlabel('Round (Day)')
ax2.set_xlabel('Round (Day)')
ax1.set_ylabel('Absolute Private Delay Cost Change')
ax2.set_ylabel('Relative Private Delay Cost Change')

# Get unique legend labels
# handles1, labels1 = ax1.get_legend_handles_labels()
# unique_labels1 = dict.fromkeys(labels1)  # Removes duplicates while preserving order
# ax1.legend(unique_labels1.keys(), loc="upper left", bbox_to_anchor=(1,1))
#ax1.legend(loc="upper left", bbox_to_anchor=(1,1))

# handles2, labels2 = ax2.get_legend_handles_labels()
# unique_labels2 = dict.fromkeys(labels2)  # Removes duplicates while preserving order
# ax2.legend(unique_labels2.keys(), loc="upper left", bbox_to_anchor=(1,1))
#ax2.legend(loc="upper left", bbox_to_anchor=(1,1))

ax1.grid(axis='y')
ax2.grid(axis='y')

# Show the combined plot
plt.tight_layout()
plt.show()