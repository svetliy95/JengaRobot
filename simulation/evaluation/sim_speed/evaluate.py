import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from multiprocessing import Process
from constants import *
import time

folder = '/home/bch_svt/cartpole/simulation/evaluation/sim_speed'

data_list = []
for ts in range(1, 12):
    for idx in range(1, 6):
        with open(folder + '/' + f'timesteps_{ts}ms_#{idx}.json', 'r') as f:
            data = json.load(f)
            mean = np.mean(data)
            rt_factor = (ts * 0.001) / mean
            data_list.append({'idx': idx, 'rt_factor': rt_factor, 'ts': ts})

df = pd.DataFrame(data_list)

ax = sns.scatterplot(data=df, x='ts', y='rt_factor', hue='idx')
ax.set(ylabel='RT-Faktor', xlabel='Zeitschritt in ms')
plt.savefig('/home/bch_svt/masterarbeit/figures/sim_speed/rt_factor_vs_timestemp.pdf', format='pdf')
plt.show()