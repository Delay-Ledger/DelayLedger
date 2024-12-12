
# %%
from datetime import datetime, timedelta

# %%
# airports and airline parameters

ops30 = ['ATL', 'BOS', 'BWI', 'CLT', 'DCA', 'DEN', 'DFW', 
            'DTW', 'EWR', 'FLL', 'HNL', 'IAD', 'IAH', 'JFK', 
            'LAS', 'LAX', 'LGA', 'MCO', 'MDW', 'MEM', 'MIA', 
            'MSP', 'ORD', 'PHL', 'PHX', 'SAN', 'SEA', 'SFO', 'SLC', 'TPA'
]

airline_color_dict = {
    'DL': 'purple', 'NK': 'yellow', 'F9': 'green', 'AA': 'red', 
    'UA': 'blue', 'B6': 'dodgerblue', 'WN': 'darkgoldenrod', 
    'G4': 'orange', 'AS':'turquoise', 'HA': 'fuchsia'
}

flt_color_dict = {
    0: 'purple', 1: 'yellow', 2: 'green', 3: 'red', 
    4: 'blue', 5: 'dodgerblue', 6: 'darkgoldenrod', 
    7: 'orange', 8:'turquoise', 9: 'fuchsia' 
}

input_csv = '682391965_T_ONTIME_MARKETING.csv'      # June
input_csv = '966008573_T_ONTIME_MARKETING.csv'      # May
input_csv = '2019_ontime/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_2019_7/On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2019_7.csv'
#input_csv = 'On_Time_Marketing_Carrier_On_Time_Performance_(Beginning_January_2018)_2019_5.csv'

# plot_date = 
crs_dep_thresh = 40         # 15 min bins
max_delay_15bin = 12         # 15 min bins
max_tot_increase_factor = 1     # 1.1, 1.2, ...

plotBool = False
runGurobi = True
timelimit = 60*15

high_priority_b = 7
low_priority_a = 3

untruthful_val = 7



# %%

    # # read data

    # df = pd.read_csv('682391965_T_ONTIME_MARKETING.csv')
    # df = df.iloc[:,:-1]     # drop last column
    # print(df.columns.tolist())
    # print('Number of flights in month: {}'.format(df.shape))

    # # read timezone data
    # df_tz = pd.read_csv('airports_tz.csv', header=None)
    # # df_tz = df_tz[(df_tz[3] == 'Canada') | (df_tz[3] == 'United States')]
    # # %%

    # # %%
    # # convert timezones to ET
    # # value is number of hours to add to local time to get ET
    # tz_dict = dict(zip(df_tz[4],df_tz[9]))
    # for key in tz_dict:
    #     try:
    #         tz_dict[key] = -5 - float(tz_dict[key])
    #     except:
    #         pass

    # # Kearney, Nebraska
    # tz_dict['EAR'] = 1


    # # %%
    # # change column headers to lowercase

    # df = df.rename(columns=str.lower)
    # df.head()



    # # %%
    # # filter for plot_date

    # # includes previous day
    # df_day = df[((df.fl_date == plot_date) | 
    #             ((df.fl_date == prev_date) & (df.dep_time > df.arr_time))) &        # previous date
    #             ((df.origin.isin(ops30) | (df.dest.isin(ops30))))]                  # origin or destination is OPS30 airport

    # # skips previous day
    # df_day = df[(df.fl_date == plot_date) &                                         # previous date
    #             ((df.origin.isin(ops30) | (df.dest.isin(ops30))))]                  # origin or destination is OPS30 airport



    # print('Number of flights on {}: {}'.format(plot_date, df_day.shape[0]))



    # # %%
    # # fix 2400 times to 0

    # df_day.loc[df_day.dep_time > 2359, 'dep_time'] = 0
    # df_day.loc[df_day.arr_time > 2359, 'arr_time'] = 0

    # # %%
    # df_day.head().T

    # # %%
    # # shift to ET

    # df_day['orig_tz_adjust'] = df_day['origin'].map(tz_dict)*4      # 15min bins
    # df_day['dest_tz_adjust'] = df_day['dest'].map(tz_dict)*4        # 15min bins


    # # %%
    # df_day.head().T
    # # %%
    # # create columns for actual departure/arrival hour

    # df_day['dep_hour'] = (df_day['dep_time'] // 100)
    # df_day['arr_hour'] = (df_day['arr_time'] // 100)

    # df_day['dep_15min'] = np.round((df_day['dep_time']-df_day['dep_hour']*100)/15)         # round, not //
    # df_day['arr_15min'] = np.round((df_day['arr_time']-df_day['arr_hour']*100)/15)

    # df_day['dep_15bin'] = df_day.dep_hour*4 + df_day.dep_15min
    # df_day['arr_15bin'] = df_day.arr_hour*4 + df_day.arr_15min

    # # create columns for departure/arrival hour

    # df_day['crs_dep_hour'] = (df_day['crs_dep_time'] // 100)
    # df_day['crs_arr_hour'] = (df_day['crs_arr_time'] // 100)

    # df_day['crs_dep_15min'] = np.round((df_day['crs_dep_time']-df_day['crs_dep_hour']*100) / 15)
    # df_day['crs_arr_15min'] = np.round((df_day['crs_arr_time']-df_day['crs_arr_hour']*100) / 15)

    # df_day['crs_dep_15bin'] = df_day.crs_dep_hour*4 + df_day.crs_dep_15min
    # df_day['crs_arr_15bin'] = df_day.crs_arr_hour*4 + df_day.crs_arr_15min

    # # adjust timezones to ET

    # for col in ['crs_dep_15bin','dep_15bin']:
    #     df_day[col] += df_day.orig_tz_adjust

    # for col in ['crs_arr_15bin','arr_15bin']:
    #     df_day[col] += df_day.dest_tz_adjust

    # # adjust overnight flights
    # for col in ['crs_arr_15bin','arr_15bin']:
    #     df_day.loc[df_day.arr_15bin < df_day.dep_15bin, col] += 96         # 96 15-min bins in a day

    # # %%
    # # filter by time

    # df_day = df_day[df_day.crs_dep_15bin < crs_dep_thresh]
    # print('flights remaining after filtering by time: {}'.format(df_day.shape[0]))

    # %%

    # if plotBool:
    #     # %%
    #     # filter for NAS delayed flights on plot_date in Ops30

    #     df1 = df_day[(df.arr_del15 == 1) & (df.nas_delay > 0)]
    #     print('Number of flights in Ops30 on {}: {}'.format(plot_date, df1.shape[0]))


    #     # %%
    #     # groupby airport, hour, carrier for both departures and arrivals

    #     grp_orig = df1.groupby(['origin','dep_hour','mkt_carrier'])
    #     df_orig = grp_orig['nas_delay'].sum().reset_index()
    #     df_orig = df_orig[df_orig.origin.isin(ops30)]
    #     df_orig.rename(columns={'origin':'apt', 'dep_hour':'hour'}, inplace=True)

    #     grp_dest = df1.groupby(['dest','arr_hour','mkt_carrier'])
    #     df_dest = grp_dest['nas_delay'].sum().reset_index()
    #     df_dest = df_dest[df_dest.dest.isin(ops30)]
    #     df_dest.rename(columns={'dest':'apt', 'arr_hour':'hour'}, inplace=True)

    #     # %%
    #     # code to print groups

    #     # for key, item in grp_orig:
    #     #     print(grp_orig.get_group(key), "\n\n")
    #     # %%

    #     # %%
    #     # combine departue and arrival delay

    #     df_delay = pd.concat([df_orig, df_dest]).groupby(by=['apt', 'hour', 'mkt_carrier']).sum().reset_index()
    #     df_delay.head()

    #     # %%
    #     # get airport & airline pairs to plot
    #     ls_apt_air = df_delay[['apt','mkt_carrier']].drop_duplicates().values.tolist()

    #     # %%
    #     df_day[df_day.dep_hour == 24].T


    #     # %%
    #     # initialize figure

    #     num_rows = 6
    #     num_cols = 5
    #     fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=ops30, horizontal_spacing=0.025, vertical_spacing=0.07)

    #     # %%
    #     # Plotly: make line chart for each airline's NAS delay at each airport

    #     row_num = 1
    #     col_num = 1
    #     ls_air_in_legend = []

    #     # loop through airports
    #     for apt in ops30:
    #         print('\n---------------')
    #         print('Airport: {}'.format(apt))
    #         airlines = [x[1] for x in ls_apt_air if apt in x]

    #         for air in airlines:
    #             delays = np.zeros(24)
    #             hours = df_delay[(df_delay.apt == apt) & (df_delay.mkt_carrier == air)].hour.values.tolist()
    #             values = df_delay[(df_delay.apt == apt) & (df_delay.mkt_carrier == air)].nas_delay.values.tolist()
    #             for idx,delay in enumerate(values):
    #                 delays[int(hours[idx])] += delay
    #             if sum(delays) > 0:
    #                 print('Airline: {}'.format(air))
    #                 fig.add_trace(go.Scatter(x=list(range(24)), y=delays,
    #                             name=air, legendgroup=air,
    #                             line=dict(color=airline_color_dict[air])),
    #                 row=row_num, col=col_num)

    #         names = set()
    #         fig.for_each_trace(
    #             lambda trace:
    #                 trace.update(showlegend=False)
    #                 if (trace.name in names) else names.add(trace.name))
                    
                    

    #         if col_num < num_cols:
    #             col_num += 1
    #         else:
    #             row_num += 1
    #             col_num = 1

    #     # %%
    #     # title and labels

    #     fig['layout']['yaxis11']['title'] = 'NAS Delay (min)'
    #     fig['layout']['xaxis28']['title'] = 'Local Time (hour)'

    #     fig.update_layout(
    #         title='NAS Delay by Airline at OPS30 airports on {}'.format(plot_date)
    #     )

    #     fig.write_html('figures/{}-nas_delay.html'.format(plot_date))
    #     fig.show()


    #     # %%
    #     ### Figure 2: Stacked Bar Chart of Airline Takeoffs+Landings per 15-minute bin at OPS30 airports

    #     # %%
    #     # groupby airport, hour, carrier for both departures and arrivals

    #     df_orig = df_day[df_day.origin.isin(ops30)]
    #     df_orig = df_orig.groupby(['origin','dep_15bin','mkt_carrier']).size().reset_index(name='count')
    #     df_orig.rename(columns={'origin':'apt', 'dep_15bin':'15bin'}, inplace=True)

    #     df_dest = df_day[df_day.dest.isin(ops30)]
    #     df_dest = df_dest.groupby(['dest','arr_15bin','mkt_carrier']).size().reset_index(name='count')
    #     df_dest.rename(columns={'dest':'apt', 'arr_15bin':'15bin'}, inplace=True)

    #     # %%
    #     # combine departure and arrival ops

    #     df_ops = pd.concat([df_orig, df_dest]).groupby(by=['apt', '15bin', 'mkt_carrier']).sum().reset_index()
    #     df_ops.head()


    #     # %%
    #     # Plotly: make stacked bar chart for each airline's number of operations at each airport

    #     fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=ops30, horizontal_spacing=0.025, vertical_spacing=0.07)

    #     row_num = 1
    #     col_num = 1
    #     ls_air_in_legend = []

    #     # loop through airports
    #     for apt in ops30:
    #         print('\n---------------')
    #         print('Airport: {}'.format(apt))
    #         airlines = [x[1] for x in ls_apt_air if apt in x]

    #         for air in airlines:
    #             ops = np.zeros(96)
    #             bins = df_ops[(df_ops.apt == apt) & (df_ops.mkt_carrier == air)]['15bin'].values.tolist()
    #             values = df_ops[(df_ops.apt == apt) & (df_ops.mkt_carrier == air)]['count'].values.tolist()
    #             for idx,delay in enumerate(values):
    #                 ops[int(bins[idx])] += delay
    #             if sum(ops) > 0:
    #                 print('Airline: {}'.format(air))
    #                 fig.add_trace(go.Bar(x=list(range(96)), y=ops,
    #                             name=air, legendgroup=air,
    #                             marker_color=airline_color_dict[air]),
    #                 row=row_num, col=col_num)

    #         names = set()
    #         fig.for_each_trace(
    #             lambda trace:
    #                 trace.update(showlegend=False)
    #                 if (trace.name in names) else names.add(trace.name))
                    
                    

    #         if col_num < num_cols:
    #             col_num += 1
    #         else:
    #             row_num += 1
    #             col_num = 1



    #     # %%
    #     # title and labels

    #     fig['layout']['yaxis11']['title'] = 'Takeoffs + Landings'
    #     fig['layout']['xaxis28']['title'] = 'Local Time (15 min)'

    #     fig.update_layout(
    #         title='Takeoffs and Landings by Airline at OPS30 airports on {}'.format(plot_date),
    #         barmode='stack'
    #     )

    #     fig.write_html('figures/{}-ops.html'.format(plot_date))
    #     fig.show()

    #     # %%


    #     # %%
    #     # Explore DTW and EWR delays for AA and UA

    #     df_day
    #     # %%
    #     df_dfw_aa = df_day[((df_day.origin == 'DFW') | (df_day.dest == 'DFW')) & (df_day.mkt_carrier == 'AA')]
    #     df_dfw_ua = df_day[((df_day.origin == 'DFW') | (df_day.dest == 'DFW')) & (df_day.mkt_carrier == 'UA')]

    #     # %%
    #     df_dfw_ua.nas_delay.sum()
    #     df_dfw_ua.shape
    #     # %%
    #     df_day.head().T
    #     # %%
    #     cols = ['mkt_carrier','op_carrier','tail_num','origin','dest','crs_dep_time','dep_time','wheels_off',
    #             'crs_arr_time','arr_time','wheels_on','arr_delay','nas_delay', 'num_flts_after']

    #     df_dfw_ua[cols]
    #     # %%
    #     df_dfw_ua[df_dfw_ua.dep_hour == 20][cols]
    #     # %%
    #     df1 = df_dfw_aa[df_dfw_aa.nas_delay>15][cols].sort_values('dep_time')
    #     df1 = df1.append(df_dfw_ua[cols])
    #     # %%
    #     df_dfw_dep = df1[df1.origin == 'DFW'].sort_values('wheels_off')
    #     df_dfw_arr = df1[df1.dest == 'DFW'].sort_values('wheels_on')
    #     # %%
    #     df_delay
    # # %%

    # #[blank]

    # # %%

    # #[blank]

    # # %%

    # # %%
    # # Create variables for MAGHP
    # """


    # ct: minimum connection time between flight i and j operated by same tail
    # cxns: flights i and j operated by same tail
    # """


    # # %%
    # # filter out cancelled and flights without tail numbers
    # df1 = df_day[~(df_day.cancelled == 1) & ~(df_day.diverted == 1) & ~(df_day.tail_num.isna())]
    # print('after filtering out cancelled flights and flights without tail, {} flights remain'.format(df1.shape[0]))

    # # filter out excessively delayed flights
    # df1 = df1[~(df1.nas_delay > 400)]
    # print('after filtering out excessively nas delayed flights, {} flights remain'.format(df1.shape[0]))

    # # %%
    # df1.to_csv('df1.csv')
    # # sys.exit()

