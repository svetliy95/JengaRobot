import json
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
matplotlib.use('TkAgg')
import pandas as pd
import json
from matplotlib import rc
import numpy as np

sns.set_theme(style="whitegrid")
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=14)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')




path = '/home/bch_svt/cartpole/simulation/evaluation/localization_speed/localization_speed1.json'

data = None
with open(path, 'r') as f:
    data = json.load(f)

# read localization speeds
with open('./detection_times.txt', 'r') as f:
    detection_times = f.readlines()
    detection_times = list(map(float, detection_times))

print(f"Data: {data}")

mean_detection_time = np.mean(detection_times)

list_of_dicts = []
for each in data:
    list_of_dicts.append({'Operation': 'Kam1', 'Dauer': each[0]})
    list_of_dicts.append({'Operation': 'Kam2', 'Dauer': each[1]})

for each in detection_times:
    print(f"Each: {each}")
    list_of_dicts.append({'Operation': 'Detektion', 'Dauer': each*2})

for each in data:
    list_of_dicts.append({'Operation': 'Transformation', 'Dauer': each[2]-mean_detection_time*2})








df = pd.DataFrame(list_of_dicts)

cam1_mean_duration = df[df['Operation'] == 'Kam1'].mean()[0]
cam2_mean_duration = df[df['Operation'] == 'Kam2'].mean()[0]
detection_mean_duration = df[df['Operation'] == 'Detektion'].mean()[0]
transformation_mean_duration = df[df['Operation'] == 'Transformation'].mean()[0]

print(f"Cam1 duration: {cam1_mean_duration + cam2_mean_duration + transformation_mean_duration + detection_mean_duration}")


fig_dims = (5.8, 3)
fig, ax = plt.subplots(figsize=fig_dims)


ax = sns.barplot(data=df, y='Operation', x='Dauer', orient='h')
# ax.set(ylabel='Achse', xlabel='Fehler in mm')
for item in ax.get_yticklabels():
    item.set_rotation(45)
plt.tight_layout()
plt.show()

