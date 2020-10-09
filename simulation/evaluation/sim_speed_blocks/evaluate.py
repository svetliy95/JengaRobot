import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from multiprocessing import Process
from constants import *
import time

folder = '/home/bch_svt/cartpole/simulation/evaluation/sim_speed_blocks'

data_list = []
ts = 5
for blocks in range(6, 54, 6):
    for idx in range(1, 11):
        try:
            with open(folder + '/' + f'timesteps_{blocks}blocks_#{idx}.json', 'r') as f:
                data = json.load(f)
                mean = np.mean(data)
                rt_factor = (ts * 0.001) / mean
                data_list.append({'idx': idx, 'rt_factor': rt_factor, 'blocks': blocks})
        except:
            pass

df = pd.DataFrame(data_list)

# ax = sns.scatterplot(data=df, x='ts', y='rt_factor', hue='idx')
ax = sns.barplot(data=df, x='blocks', y='rt_factor', estimator=np.mean)
ax.set(ylabel='RT-Faktor', xlabel='Zeitschritt in ms')
plt.savefig('/home/bch_svt/masterarbeit/figures/sim_speed/rt_factor_vs_timestemp.pdf', format='pdf')
plt.show()