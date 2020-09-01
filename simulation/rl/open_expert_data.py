import numpy as np
import math


np.set_printoptions(suppress=True, precision=5)

data = np.load('/home/bch_svt/cartpole/simulation/rl/expert_data21-08-2020_13:32:00.npz', allow_pickle=True)
for key in data.keys():
    print(key)

rewards = data['rewards']
print(rewards)

# count extracted blocks
extracted_blocks = 0
for r in rewards:
    if math.isclose(r, 4):
        extracted_blocks += 1
print(extracted_blocks)

observations = data['obs']
for o in observations:
    print(repr(o).replace('\n', ''))
