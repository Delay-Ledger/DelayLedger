import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta, date
from utils import color_boxplot, create_piecewise_function, expected_pv, cost_jump_arrivals
from params import *

# boxplot
run_type = 'intra-alt-intra' 
folder_name = 'test_stochastic_not_mvp'

subdir_name = folder_name + '/' + run_type

subdir_full_path = os.getcwd()+'/'+subdir_name

df_delays = pd.read_csv(subdir_full_path+'/df_delays.csv')

this_date = date(2019, 5, 1)
end_date = date(2019, 5, 30)
delta = timedelta(days=1)

coordinating_airline = pd.read_csv(subdir_full_path+'/coordinating_airline.csv',header=None)[0]
i=0
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))

while this_date <= end_date:
    coord_data_baseline = df_delays[df_delays["marketing_airline_network"] == coordinating_airline[i]][this_date.strftime("%Y-%m-%d")+'-Baseline-Weighted'].dropna()
    participant_data_baseline = df_delays[df_delays["marketing_airline_network"] != coordinating_airline[i]][this_date.strftime("%Y-%m-%d")+'-Baseline-Weighted'].dropna()

    coord_data_deled = df_delays[df_delays["marketing_airline_network"] == coordinating_airline[i]][this_date.strftime("%Y-%m-%d")+'-'+coordinating_airline[i]+'-Intra2-Weighted'].dropna()
    participant_data_deled = df_delays[df_delays["marketing_airline_network"] != coordinating_airline[i]][this_date.strftime("%Y-%m-%d")+'-'+coordinating_airline[i]+'-Intra2-Weighted'].dropna()

    coord_delay_change = coord_data_deled - coord_data_baseline
    participant_delay_change = participant_data_deled - participant_data_baseline

    color_boxplot(data=participant_delay_change, ax=ax1, pos=[i+1])
    ax1.scatter([i+1] * len(coord_delay_change), coord_delay_change, color=airline_color_dict[coordinating_airline[i]], label = coordinating_airline[i], marker='o', zorder=5)

    coord_percentage_change = (coord_data_deled - coord_data_baseline) / (coord_data_baseline) * 100
    participant_percentage_change = (participant_data_deled - participant_data_baseline) / (participant_data_baseline) * 100

    color_boxplot(data=participant_percentage_change, ax=ax2, pos=[i+1])
    ax2.scatter([i+1] * len(coord_percentage_change), coord_percentage_change, color = airline_color_dict[coordinating_airline[i]], label=coordinating_airline[i], marker='o', zorder=5)

    this_date += delta
    i += 1

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