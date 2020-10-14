import json
from utils.utils import remove_outliers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib; matplotlib.use("TkAgg")
import csv
from matplotlib import rc

sns.set_theme(style="whitegrid")
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=14)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')


def get_from_logging_file():
    data = {'x': [], 'y': [], 'z': [], 'Mx': [], 'My': [], 'Mz': []}
    with open('./forces_2ms.fsl') as f:
        reader = csv.reader(f)
        for i, l in enumerate(reader):
            if i > 9:
                data['x'].append(float(l[1]))
                data['y'].append(float(l[2]))
                data['z'].append(float(l[3]))
                data['Mx'].append(float(l[4]))
                data['My'].append(float(l[5]))
                data['Mz'].append(float(l[6]))

    return data

def create_data_frame_from_file(avg_time):
    data = []
    with open(f'./forces_{avg_time}ms.fsl') as f:
        reader = csv.reader(f)
        for i, l in enumerate(reader):
            if i > 9:
                data.append({'axis': 'Fx', 'value': float(l[1]), 'avg_time': avg_time})
                data.append({'axis': 'Fy', 'value': float(l[2]), 'avg_time': avg_time})
                data.append({'axis': 'Fz', 'value': float(l[3]), 'avg_time': avg_time})
                data.append({'axis': 'Mx', 'value': float(l[4]), 'avg_time': avg_time})
                data.append({'axis': 'My', 'value': float(l[5]), 'avg_time': avg_time})
                data.append({'axis': 'Mz', 'value': float(l[6]), 'avg_time': avg_time})
    df = pd.DataFrame(data)

    return df


df1 = create_data_frame_from_file(2)
df2 = create_data_frame_from_file(10)
df3 = create_data_frame_from_file(50)
df4 = create_data_frame_from_file(100)
df5 = create_data_frame_from_file(200)
df6 = create_data_frame_from_file(500)

dframes = [df1, df2, df3, df4, df5, df6]

df = pd.concat(dframes)
x_axis = df['axis'] == "Fx"
avg_time_2ms = df['avg_time'] == 2
df_x = df[x_axis]

print(f"2ms std_dev: {df_x['value'].std()}")

print(f"Df:\n {df}")

# sns.lineplot(data=df_x, x=df_x.index, y='value')
#
fig_dims = (5.6, 3)
fig, ax = plt.subplots(figsize=fig_dims)

ax = sns.barplot(data=df_x, x='value', y='avg_time', orient='h', estimator=np.std)
ax.set(ylabel='Zeitkonstante', xlabel='Standardabweichung')
plt.tight_layout()
plt.show()

