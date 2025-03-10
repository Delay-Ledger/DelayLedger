import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('test2/intra-alt-intra/df_delays.csv')

def color_boxplot(data, ax, pos, vert=True, color='k', widths=0.5):
    ax = ax or plt.gca()
    bp = ax.boxplot(data, patch_artist=True, showmeans=False, positions=pos, whis=1000, widths=widths, vert=vert)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[item], color=color)
    for patch in bp['boxes']:
        patch.set(facecolor='w', linewidth=2)
    for item in ['caps', 'fliers']:
        plt.setp(bp[item], linewidth=2)
    for item in ['whiskers']:
        plt.setp(bp[item], linewidth=2)
    for item in ['medians']:
        plt.setp(bp[item], linewidth=2)


aa_data_0501 = df[df["marketing_airline_network"] == "AA"]["2019-05-01-Intra1"].dropna()
non_aa_data_0501 = df[df["marketing_airline_network"] != "AA"]["2019-05-01-Intra1"].dropna()
nk_data_0502 = df[df["marketing_airline_network"] == "NK"]["2019-05-02-Intra1"].dropna()
non_nk_data_0502 = df[df["marketing_airline_network"] != "NK"]["2019-05-02-Intra1"].dropna()
fig, ax = plt.subplots(figsize=(10, 6))
color_boxplot(data=non_aa_data_0501, ax=ax, pos=[1], vert=True, color='k', widths=0.5)
ax.scatter([1] * len(aa_data_0501), aa_data_0501, color='red', marker='o', label='American', zorder=5)

color_boxplot(data=non_nk_data_0502, ax=ax, pos=[2], vert=True, color='k', widths=0.5)
ax.scatter([2] * len(nk_data_0502), nk_data_0502, color='yellow', marker='o', label='Spirit', zorder=5)

ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.invert_yaxis()
ax.set_xticks([1, 2])
ax.legend()

plt.tight_layout()
plt.show()

color_boxplot(data=df,ax=ax,pos=[2],vert=True, color='k', widths=0.5)
plt.tight_layout()
plt.show()
