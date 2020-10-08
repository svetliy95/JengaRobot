import json
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd



path = '/home/bch_svt/cartpole/simulation/evaluation/localization_speed/localization_speed1.json'

data = None
with open(path, 'r') as f:
    data = json.load(f)

list_of_dicts = []
for each in data:
    list_of_dicts.append({'what': 'cam1', 'time': each[0]})
    list_of_dicts.append({'what': 'cam2', 'time': each[1]})
    list_of_dicts.append({'what': 'transformation', 'time': each[2]})

df = pd.DataFrame(list_of_dicts)

sns.boxplot(data=df, x='what', y='time')

plt.show()

