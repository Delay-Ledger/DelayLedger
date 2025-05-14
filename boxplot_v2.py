# %%
from cProfile import run
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib.ticker import FormatStrFormatter
plt.rcParams["font.family"] = "Times New Roman"
SMALL_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


import numpy as np
from datetime import datetime, timedelta, date
import os
from utils import record_changes, color_boxplot

# %%
def collect_airline_effect_data(df_dlm, plot_type):

    # loop through dates and record delay changes per airline and control
    this_date = date(2019, 5, 1)
    end_date = date(2019, 5, 30) # CHANGE:
    delta = timedelta(days=1)

    # determine which airline was in control
    plot_type_ls = ['IA-DLM-IA', 'DLM-IA']
    order_control = {a:[] for a in plot_type_ls}
    airline_diff = {a:{} for a in plot_type_ls}
    

    while this_date <= end_date:
        str_date = this_date.strftime("%Y-%m-%d")

        # dlm-intra-dlm
        # create col of difference between post-Intra2 and post-Intra1
        intra2_col = [c for c in df_dlm.columns if str_date in c and '-Intra2' in c and 'Weighted' not in c][0]
        intra2_col_weighted = [c for c in df_dlm.columns if str_date in c and '-Intra2-Weighted' in c][0]

        df_dlm[str_date+'-Intra2-Intra1'] = (df_dlm[intra2_col] - df_dlm[str_date+'-Intra1'])#/df_dlm[str_date+'-Intra1']
        df_dlm[str_date+'-Intra2-Intra1-Weighted'] = (df_dlm[intra2_col_weighted] - df_dlm[str_date+'-Intra1-Weighted'])#/df_dlm[str_date+'-Intra1-Weighted']
        df_dlm.fillna(0, inplace=True)
        if plot_type == 'percent':
            df_dlm[str_date+'-Intra2-Intra1-Weighted'] = df_dlm[str_date+'-Intra2-Intra1-Weighted']/df_dlm[str_date+'-Intra1-Weighted']
        
        this_date += delta

        df_ls = [df_dlm]

        for df,dict_key in zip(df_ls,plot_type_ls):

            cols = [c for c in df.columns if str_date in c and 'avg' in c and 'Baseline' not in c and 'Intra1' not in c and 'incr' not in c]
            # print(cols)
            airline = cols[0][-6:-4]
            print(airline, 'in control')
            order_control[dict_key].append(airline)

            
            diff_cols = [c for c in df.columns if '-Intra2-Intra1-Weighted' in c]    

            for airline in df.index:
                print(airline)
                airline_diff[dict_key][airline] = df.loc[airline][diff_cols].tolist()

    return order_control, airline_diff


# %%
# parameters
airline_color = {
    'AA': 'dimgrey',
    'UA': 'deepskyblue',
    'DL': '#c8102e',
    'WN': 'darkorange',
    'B6': 'mediumspringgreen',
    'AS': 'blue',
    'NK': 'gold',
    'F9': 'seagreen',
    # 'G4': 'red',
    
}

airline_symbols = {
    'AA': 'v',
    'AS': '^',
    'B6': '>',
    'DL': '<',
    'F9': 'o',
    'NK': 'X',
    'UA': 's',
    'WN': 'H'
}

# folder_name = 'exp30_fixed_low_priority_bug'
# folder_name = 'exp30_ledger_past_5_days'
# folder_name = 'exp30_ledger_wrt_intra_30'
# folder_name = 'exp30_no_repeats_30'
# folder_name = 'exp30_no_repeats_40'
# folder_name = 'exp30_ledger_wrt_intra_40'
# folder_name = 'exp30_may'
# folder_name = 'exp30_may_tot_delaycant_incr'
# folder_name = 'exp30_may_tot_delay_cant_incr_no_cxn_bonus'
# folder_name = 'test30'
# folder_name = 'standard_decisions_eval_stochastic3'

# folder_name = 'standard_decisions_eval_stochastic_lambda5_surge10
# folder_name = 'eval_new_piecewise_30days_fixed_priorities'
# folder_name = 'eval_new_piecewise_30days'
# folder_name = 'standard_decisions_eval_mvf_lambda5_surge10'

# folder_name = 'standard_decisions_eval_mvf_lambda5_surge10_test30'


# folder_name = 'stochastic_lambda5_surge10_eval_mvf_test30'
# folder_name = 'standard_decisions_eval_standard_30days'
folder_name = 'standard_decisions_eval_mvf_lambda5_surge10_test30'

# CHANGE:

if not os.path.exists(folder_name+'/figures'):
    os.mkdir(folder_name+'/figures')

# %%

# count number of flights
# Re-process data
this_date = date(2019, 5, 1)
end_date = date(2019, 5, 30) # CHANGE:
delta = timedelta(days=1)
subdir_full_path = folder_name + '/intra-alt-intra'
str_date = this_date.strftime("%Y-%m-%d")

num_flts_ls = []
airline_flts_dict = {k:0 for k in airline_color.keys()}
priority_dict = {k:0 for k in ['high','medium','low']}

while this_date <= end_date:
    # string
    str_date = this_date.strftime("%Y-%m-%d")
    print(str_date)

    df = pd.read_csv(subdir_full_path+'/'+str_date+'/df1.csv')

    # number of flights
    num_flts_ls.append(df.shape[0])

    for airline in airline_flts_dict:
        airline_flts_dict[airline] += df[df.marketing_airline_network == airline].shape[0]

    # high
    priority_dict['high'] += df[df.flight_val >= 7].shape[0]
    priority_dict['medium'] += df[(df.flight_val > 3) & (df.flight_val < 7)].shape[0]
    priority_dict['low'] += df[df.flight_val <= 3].shape[0]

    this_date += delta

print('avg:', np.mean(num_flts_ls))
print(airline_flts_dict)
print(priority_dict)

# %%

# pie charts
f, ax1 = plt.subplots(1, 1, figsize=(3,3))
ax1.pie(airline_flts_dict.values(), labels=airline_flts_dict.keys(), colors=airline_color.values(),
    autopct='%.1f%%', pctdistance=1.25, labeldistance=None, startangle=0, textprops={'fontsize': 14})
# plt.legend()
plt.savefig(folder_name+'/figures/marketshare.png', facecolor='w',dpi=600, bbox_inches='tight')

# ax2.pie(priority_dict.values(), labels=priority_dict.keys(),
#     autopct='%.1f%%', pctdistance=1.25, labeldistance=None, startangle=0)

# %%
# read intra-alt-intra
df_iai = pd.read_csv(folder_name+'/intra-alt-intra/df_delays.csv', index_col=0)
delay_cols = [c for c in df_iai.columns if '-Intra2' in c and 'Weighted' not in c]
delay_weighted_cols = [c for c in df_iai.columns if '-Intra2-Weighted' in c]
# print(delay_cols)
# print(delay_weighted_cols)

iai_delay = [15*x for x in df_iai[delay_cols].sum().tolist()]
iai_weighted_delay = [15*x for x in df_iai[delay_weighted_cols].sum().tolist()]
print(len(iai_delay))

# read baseline
df_iai.columns.tolist()
delay_cols = [c for c in df_iai.columns if '-Baseline' in c and not any(x in c for x in ['avg','Weighted'])]
delay_weighted_cols = [c for c in df_iai.columns if '-Baseline-Weighted' in c]

baseline_delay = [15*x for x in df_iai[delay_cols].sum().tolist()]
baseline_weighted_delay = [15*x for x in df_iai[delay_weighted_cols].sum().tolist()]

print(len(baseline_delay))

# read just-intra
delay_cols = [c for c in df_iai.columns if '-Intra1' in c and 'Weighted' not in c and 'avg' not in c]
delay_weighted_cols = [c for c in df_iai.columns if '-Intra1-Weighted' in c and 'avg' not in c]

intra_delay = [15*x for x in df_iai[delay_cols].sum().tolist()]
intra_weighted_delay = [15*x for x in df_iai[delay_weighted_cols].sum().tolist()]

print(len(intra_delay))

# %%
# average percent change
baseline_weighted_delay
intra_weighted_delay
iai_weighted_delay

# print('Average reduction from Intra is:', np.mean(np.subtract(baseline_weighted_delay, intra_weighted_delay)/baseline_weighted_delay))
# print('Average reduction from DLM is:', np.mean(np.subtract(intra_weighted_delay, iai_weighted_delay)/intra_weighted_delay))


# %%

# %%
# read data
# df_dlm = pd.read_csv(folder_name+'/alt-intra/df_delays.csv', index_col=0)
df_dlm = pd.read_csv(folder_name+'/intra-alt-intra/df_delays.csv', index_col=0)
run_type = 'ia-dlm-ia'
# run_type = 'dlm-ia'
plot_type = 'total'             # total or percent
# plot_type = 'percent'

# %%
df_dlm.columns.tolist()




# %%
order_control, airline_diff_perc = collect_airline_effect_data(df_dlm, 'percent')
_, airline_diff_mean = collect_airline_effect_data(df_dlm, 'mean')

# %%
run_type = 'IA-DLM-IA'
# run_type = 'DLM-IA'

controlling_ls_perc = []
controlling_ls_mean = []
participate_ls_perc = []
participate_ls_mean = []


for idx,a in enumerate(order_control[run_type]):
    # controlling airline
    controlling_ls_perc.append(airline_diff_perc[run_type][a][idx])
    controlling_ls_mean.append(airline_diff_mean[run_type][a][idx])

    # participating airlines
    other_airlines_perc = []
    other_airlines_mean = []
    for b in airline_color:
        if b != a:
            other_airlines_perc.append(airline_diff_perc[run_type][b][idx])
            other_airlines_mean.append(airline_diff_mean[run_type][b][idx])
    participate_ls_perc.append(other_airlines_perc)
    participate_ls_mean.append(other_airlines_mean)

# %%
participate_ls_mean

# %%

df_dlm_mean = pd.DataFrame(airline_diff_mean['IA-DLM-IA']).T
df_dlm_mean.to_csv(folder_name+'/df_dlm_mean.csv')

df_dlm_perc = pd.DataFrame(airline_diff_perc['IA-DLM-IA']).T
df_dlm_perc.to_csv(folder_name+'/df_dlm_perc.csv')

# %%
## Save order of control
with open(folder_name+"/order_control.json", 'w') as f:
    # indent=2 is not needed but makes the file human-readable 
    # if the data is nested
    json.dump(order_control, f, indent=2) 

# with open("file.json", 'r') as f:
#     order_control = json.load(f)

# %%
### Boxplot comparison

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(18,12))
days = range(1,len(controlling_ls_perc)+1)
print(len(days))
print(len(controlling_ls_mean))

# ax1
# ax1.yaxis.grid(color='k', alpha=0.5)
ax1.set_facecolor('#EAEAF2')
ax1.xaxis.grid(color='white')
ax1.yaxis.grid(color='white')

for x, control, participants, a in zip(days, controlling_ls_mean, participate_ls_mean, order_control[run_type]):
    ax1.axhline(y=0, color='k', alpha=0.5)
    color_boxplot(participants, ax1, [x])
    ax1.scatter(x, control, color=airline_color[a],marker='o',s=300,zorder=10,edgecolor='k')
# ax1.set_ylabel('Private Delay Cost Change',fontsize=25)
vals = ax1.get_yticks();
ax1.set_yticklabels(vals, fontsize=24);
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines["left"].set_visible(True)
ax1.spines["bottom"].set_visible(True)

# ax2
ax2.set_facecolor('#EAEAF2')
ax2.xaxis.grid(color='white')
ax2.yaxis.grid(color='white')
for x, control, participants, a in zip(days, controlling_ls_perc, participate_ls_perc, order_control[run_type]):
    ax2.axhline(y=0, color='k', alpha=0.5)
    color_boxplot(participants, ax2, [x])
    ax2.scatter(x, control, color=airline_color[a],marker='o',s=300,zorder=10,edgecolor='k')
# ax2.set_ylabel('Percentage Private Delay Cost Change',fontsize=25)
# ax2.set_xlabel('Round (Day)',fontsize=25)
vals = ax2.get_yticks();
ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals], fontsize=24);

vals = ax2.get_xticks();
ax2.set_xticklabels(vals, fontsize=24)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines["left"].set_visible(True)
ax2.spines["bottom"].set_visible(True)

# legend
# markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='', ms=15) for color in airline_color.values()]
# plt.legend(markers, airline_color.keys(), labelspacing=2, bbox_to_anchor=(1.2, 0.8, 0.3, 0.2),prop={'size': 20})
# plt.legend(markers, airline_color.keys(), bbox_to_anchor=(-0.12, 2.2, 0.3, 0.2), 
#     numpoints=1,loc="lower left",prop={'size': 25},) #ncol=len(airline_color)


# ax2
# ax2.scatter(x, controlling_ls_perc, marker='s')
# ax2.set_ylabel('Private Delay Cost Change')
# ax2.set_xlabel('Round (day)')
# vals = ax2.get_yticks();
# ax2.set_yticklabels(['{:,.0%}'.format(x) for x in vals]);

plt.savefig(folder_name+'/figures/airline_comparisons_'+run_type+'.png', facecolor='w', dpi=600, bbox_inches='tight')


# %%

# create stand-alone legend
legend_entries = ['American (1)', 'United (3)', 'Delta (4)', 'Southwest (4)', 'JetBlue (5)', 'Alaska (3)', 'Spirit (4)', 'Frontier (6)']

markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', markeredgecolor='k', linestyle='', ms=15) for color in airline_color.values()]
leg = plt.legend(markers, airline_color.keys(), facecolor='white', labelspacing=1, bbox_to_anchor=(1.3, 0.8, 0.3, 0.2),prop={'size': 16},edgecolor='k', ncol=2)
# leg.set_title('Coordinator (*)',prop={'size':18})
plt.savefig(folder_name+'/figures/airline_plain_legend.png', facecolor='w', dpi=600, bbox_inches='tight')


#######
# Proportion of Total Flights
#######

total_flights = sum(airline_flts_dict.values())
prop_flights = {k : np.round((100 * v / total_flights),1) for k, v in airline_flts_dict.items()}
print('Proportion of Total Flights:', prop_flights)

#######
# Public delay
#######

# read intra-alt-intra
df_iai = pd.read_csv(folder_name+'/intra-alt-intra/df_delays.csv', index_col=0)

public_delay_intra2_cols = [c for c in df_iai.columns if '-Intra2' in c and 'Weighted' not in c]

public_delay_intra1_cols = [c for c in df_iai.columns if '-Intra1' in c and 'Weighted' not in c and 'avg' not in c]

# change in public delay
# public_delay_intra2 - public_delay_intra1

public_delay_intra2 = df_iai[public_delay_intra2_cols]
# rename columns to only dates
public_delay_intra2.columns = public_delay_intra2.columns.str.split('-').str[:3].str.join('-')

public_delay_intra1 = df_iai[public_delay_intra1_cols]
# rename columns to only dates
public_delay_intra1.columns = public_delay_intra1.columns.str.split('-').str[:3].str.join('-')

public_delay_change = np.round(((public_delay_intra2 - public_delay_intra1)/public_delay_intra1*100).T.mean(),1)

print('Average change in public delay (%)', public_delay_change)

prop_flights_series = pd.Series(prop_flights)

print('Overall average change in public delay (%)', (prop_flights_series.reindex(public_delay_change.index) * public_delay_change).sum()/100)

#######
# Overall private delay
#######

private_delay_intra2_cols = [c for c in df_iai.columns if '-Intra2-Weighted' in c]
private_delay_intra1_cols = [c for c in df_iai.columns if '-Intra1-Weighted' in c and 'avg' not in c]
# change in private delay
# private_delay_intra2 - private_delay_intra1

private_delay_intra2 = df_iai[private_delay_intra2_cols]
# rename columns to only dates
private_delay_intra2.columns = private_delay_intra2.columns.str.split('-').str[:3].str.join('-')

private_delay_intra1 = df_iai[private_delay_intra1_cols]
# rename columns to only dates
private_delay_intra1.columns = private_delay_intra1.columns.str.split('-').str[:3].str.join('-')

private_delay_change = np.round(((private_delay_intra2 - private_delay_intra1)/private_delay_intra1*100).T.mean(),1)

print('Average private cost change, Overall (%)',private_delay_change)

print('Overall average change in private delay (%)', (prop_flights_series.reindex(private_delay_change.index) * private_delay_change).sum()/100)


# #######
# # With Intra-Airline Substitution
# #######

# private_delay_baseline_cols = [c for c in df_iai.columns if '-Baseline-Weighted' in c]

# private_delay_baseline = df_iai[private_delay_baseline_cols]
# # rename columns to only dates
# private_delay_baseline.columns = private_delay_baseline.columns.str.split('-').str[:3].str.join('-')

# intra1_baseline_private_delay_change = np.round(((private_delay_intra1 - private_delay_baseline)/private_delay_baseline*100).T.mean(),1)

# print('Average private cost change, Intra-Airline Substitution (%)', intra1_baseline_private_delay_change)

# #######
# # With DeLed
# #######

# intra2_baseline_private_delay_change = np.round(((private_delay_intra2 - private_delay_baseline)/private_delay_baseline*100).T.mean(),1)

# print('Average private cost change, Intra2 vs Baseline (%)', intra2_baseline_private_delay_change)













# %%
