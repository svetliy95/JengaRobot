import json
from utils.utils import remove_outliers
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib; matplotlib.use("TkAgg")
import csv

def get_from_logging_file():
    data = {'x': [], 'y': [], 'z': [], 'Mx': [], 'My': [], 'Mz': []}
    with open('jitter_data/f777.fsl') as f:
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

with open('jitter_data/forces_r1_b2_d1_g0.json') as f:
    forces1 = json.load(f)

with open('jitter_data/forces_r1_b5_d3_g1.json') as f:
    forces2 = json.load(f)

forces_1_x = [v[0] for v in forces1]
data1 = {'x': [], 'y': [], 'z': [], 'Mx': [], 'My': [], 'Mz': []}
data2 = {'x': [], 'y': [], 'z': [], 'Mx': [], 'My': [], 'Mz': []}
for d in forces1:
    data1['x'].append(d[0])
    data1['y'].append(d[1])
    data1['z'].append(d[2])
    data1['Mx'].append(d[3])
    data1['My'].append(d[4])
    data1['Mz'].append(d[5])

for d in forces2:
    data2['x'].append(d[0])
    data2['y'].append(d[1])
    data2['z'].append(d[2])
    data2['Mx'].append(d[3])
    data2['My'].append(d[4])
    data2['Mz'].append(d[5])

data2 = get_from_logging_file()
data_to_plot = data2['x'][1000:2000]
x = np.linspace(0, len(data_to_plot), len(data_to_plot))
sns.lineplot(x=x,y=data_to_plot)
plt.show()

data = [{'id': 1, 'axis': 'x', 'std_dev': np.std(data1['x'])}, {'id': 1, 'axis': 'y', 'std_dev': np.std(data1['y'])},
        {'id': 1, 'axis': 'z', 'std_dev': np.std(data1['z'])}, {'id': 1, 'axis': 'Mx', 'std_dev': np.std(data1['Mx'])},
        {'id': 1, 'axis': 'My', 'std_dev': np.std(data1['My'])}, {'id': 1, 'axis': 'Mz', 'std_dev': np.std(data1['Mz'])},
        {'id': 2, 'axis': 'x', 'std_dev': np.std(data2['x'])}, {'id': 2, 'axis': 'y', 'std_dev': np.std(data2['y'])},
        {'id': 2, 'axis': 'z', 'std_dev': np.std(data2['z'])}, {'id': 2, 'axis': 'Mx', 'std_dev': np.std(data2['Mx'])},
        {'id': 2, 'axis': 'My', 'std_dev': np.std(data2['My'])}, {'id': 2, 'axis': 'Mz', 'std_dev': np.std(data2['Mz'])}
        ]
data = pd.DataFrame(data)

sns.catplot(x='axis', y='std_dev', hue='id', data=data, kind='bar')
plt.show()


