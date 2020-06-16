import numpy as np
import math

data = np.load('expert_data11-06-2020_20:57:41.npz', allow_pickle=True)
np.save('delete_this', data)
test = np.load('delete_this.npy')
print(test)
for key in data.keys():
    print(key)

rewards = data['rewards']

# count extracted blocks
extracted_blocks = 0
for r in rewards:
    if math.isclose(r, 4):
        extracted_blocks += 1
print(extracted_blocks)


episode_starts = data['episode_starts']
for i in episode_starts:
    print(i)
