import numpy as np
import math

data = np.load('/home/bch_svt/cartpole/simulation/rl/expert_data_30_20_toppled_layer_state_7.npz', allow_pickle=True)
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
