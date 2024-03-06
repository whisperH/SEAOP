import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.size'] = 14  # 设置全局字体大小

HCC_Label = ['X12287P','X1240T','X424T','X314P']
HCC_anomaly = [90     ,68,       67,      57]
HCC_inlier = [1       ,23,       24,      34]
HCC_score = ["***",      "***" ,    "***",  "*"]

LUAD_Label = ['LUAD.92N','LUAD.26T','LUAD.03T','LUAD.03N','LUAD.90T','LUAD.43N']
LUAD_anomaly = [90,      89,         67,        65,        57,        47,]
LUAD_inlier = [0,        0,          22,        25,        32,        43,]
LUAD_score = ["***",      "***" ,    "***",     "***",     "**",    "n"]

HeLa_label = ['H80','H83','H249','H263','H266','H17','H227','H38','H46','H70','H123', 'H179']
HeLa_anomaly = [91  , 91  , 91,    86,    86,    64,   62,    60,  56,   54,   52,    48]
HeLa_inlier = [0  ,    0  ,  0,    5,     5 ,    27,   29,    31,  35,   37,   39,    43]
HeLa_score = ["***","***" ,"***","***", "***", "***", "***", "**", "*",  "n",   "n",    "n"]

title = ["HeLa", "HCC", "LUAD"]
anomalys = [HeLa_anomaly, HCC_anomaly,LUAD_anomaly]
inliers = [HeLa_inlier, HCC_inlier,LUAD_inlier]
labels = [HeLa_label, HCC_Label,LUAD_Label]
score_values = [
    HeLa_score,
    HCC_score,
    LUAD_score,
]
widths = [ 0.8, 0.8, 0.8,]
xlabel = ['(a)', '(b)', '(c)']


fig = plt.figure(figsize=(12, 12))
gs0 = gridspec.GridSpec(2, 2, figure=fig)
fig.add_subplot(gs0[0, :])
ax2 = fig.add_subplot(gs0[1, 0])
ax3 = fig.add_subplot(gs0[1, 1])

for idx, iax in enumerate(fig.axes):
    ind = np.arange(len(labels[idx]))
    bottom = np.zeros(len(labels[idx]))
    bar_heightest = 0

    bar_plot1 = iax.bar(ind, anomalys[idx], widths[idx], color='#d47d6c', label='outlier')
    bottom += anomalys[idx]
    bar_plot2 = iax.bar(ind, inliers[idx], widths[idx],  bottom=anomalys[idx], color='#a9aaad', label='inlier')
    bottom += inliers[idx]
    for bar_idx, rect in enumerate(bar_plot2):
        if bottom[bar_idx] > bar_heightest:
            bar_heightest = bottom[bar_idx]
        iax.text(rect.get_x() + rect.get_width() / 2., 0.5 + bottom[bar_idx],
                score_values[idx][bar_idx],
                ha='center', va='bottom', rotation=0)
    iax.bar_label(bar_plot1, label_type='center', color='white', fontsize=16)
    # iax.bar_label(bar_plot2, label_type='center')
    iax.set_title(f'{xlabel[idx]}')

    iax.set_ylabel('Outlier counts')
    if idx == 0:
        iax.set_xticks(ind, labels[idx], rotation=30)
    else:
        iax.set_xticks(ind, labels[idx], rotation=30)
    iax.set_yticks(np.arange(0, 120, 20))
    # iax.set_xlabel(f'({xlabel[idx]})')
    if idx == 0:
        iax.text(-2.2, -9, title[idx], fontsize=20)
    elif idx == 1:
        iax.text(-1.3, -9, title[idx], fontsize=20)
    else:
        iax.text(-1.7, -9, title[idx], fontsize=20)
    # iax.legend()
plt.tight_layout()

# plt.show()
plt.savefig("./all_outliers.png", dpi=300)