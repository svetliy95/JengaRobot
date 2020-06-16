import os
import re
import numpy as np


folder = "."
pattern = "expert_data*"
all_filenames = os.listdir(folder)
filenames = []
for filename in all_filenames:
    if re.match(pattern, filename):
        filenames.append(filename)

expert_data = {'actions': np.reshape(np.array([]), (0, 1)),
               'obs': np.reshape(np.array([]), (0, 13)),
               'rewards': np.array([]),
               'episode_returns': np.array([]),
               'episode_starts': np.array([])}

for filename in filenames:
    data = np.load(filename)
    for key in expert_data.keys():
        print(expert_data[key].shape)
        print(data[key].shape)
        expert_data[key] = np.concatenate((expert_data[key], data[key]), 0)

np.savez('combined_expert_data', **expert_data)

