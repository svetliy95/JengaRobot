# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# import pandas as pd
# import json
# from matplotlib import rc
# import seaborn as sns
#
# sns.set_theme(style="whitegrid")
# rc('text', usetex=True)
# rc('font', family='serif')
# rc('font', size=14)
# rc('legend', fontsize=13)
# rc('text.latex', preamble=r'\usepackage{cmbright}')
#
# fig_dims = (5.8, 3.2)
# fig, ax = plt.subplots(figsize=fig_dims)
#
# ax = sns.boxplot(ax=ax, x='error', y='axis', hue='Detektor', data=df, orient='h', showfliers=False)
# # ax.set(ylabel='Achse', xlabel='Fehler in $^\circ$')
# ax.set(ylabel='Achse', xlabel='Fehler in mm')
# ax.set(xlim=(-0.01, 1.5))
# plt.tight_layout()
# # plt.savefig('/home/bch_svt/masterarbeit/figures/jitter/detection_libraries_jitter_comparison.pdf', format='pdf')
# plt.show()


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import json
from matplotlib import rc
import seaborn as sns

sns.set_theme(style="whitegrid")
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=14)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')

fig_dims = (5.8, 3.2)
fig, ax = plt.subplots(figsize=fig_dims)



ax = sns.boxplot(ax=ax, x='error', y='axis', hue='Detektor', data=df, orient='h', showfliers=False)
# ax.set(ylabel='Achse', xlabel='Fehler in $^\circ$')
ax.set(ylabel='Achse', xlabel='Fehler in mm')
ax.set(xlim=(-0.01, 1.5))
plt.tight_layout()
# plt.savefig('/home/bch_svt/masterarbeit/figures/jitter/detection_libraries_jitter_comparison.pdf', format='pdf')
plt.show()



