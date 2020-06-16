import csv
import numpy as np


# read data from file
filename = "observation_data_with_current_step.csv"
with open(filename, mode='r') as f:
    reader = csv.reader(f)
    data = []
    for row in reader:
        print(row)
        row = list(map(float, row))
        data.append(row)

# safe data feature-wise
data_dict = {}
# [force, block_distance, total_block_distance, current_step_tower_displacement[0],
#                              current_step_tower_displacement[1], current_round_displacement[0],
#                              current_round_displacement[1], last_tilt_2ax[0], last_tilt_2ax[1], current_step_z_rot,
#                              z_rotation_last, side, block_height]
data_dict['force'] = [i[0] for i in data]
data_dict['block_distance'] = [i[1] for i in data]
data_dict['total_block_distance'] = [i[2] for i in data]
data_dict['current_step_tower_displacement1'] = [i[3] for i in data]
data_dict['current_step_tower_displacement2'] = [i[4] for i in data]
data_dict['current_round_displacement1'] = [i[5] for i in data]
data_dict['current_round_displacement2'] = [i[6] for i in data]
data_dict['last_tilt_1'] = [i[7] for i in data]
data_dict['last_tilt_2'] = [i[8] for i in data]
data_dict['current_step_z_rot'] = [i[9] for i in data]
data_dict['z_rotation_last'] = [i[10] for i in data]
data_dict['side'] = [i[11] for i in data]
data_dict['block_height'] = [i[12] for i in data]
data_dict['action'] = [i[13] for i in data]
data_dict['reward'] = [i[14] for i in data]

calculated_rewards = []
for i in range(len(data)):
    block_distance = data_dict['block_distance'][i]
    tower_displacement = np.linalg.norm(np.array([data_dict['current_step_tower_displacement1'][i], data_dict['current_step_tower_displacement2'][i]]))
    tilt = np.linalg.norm(np.array([data_dict['last_tilt_1'][i], data_dict['last_tilt_2'][i]]))
    z_rot = data_dict['current_step_z_rot'][i]
    reward = data_dict['reward'][i]
    total_block_distance = data_dict['total_block_distance'][i]
    total_tower_displacement = np.linalg.norm(data_dict['current_round_displacement1'][i] + data_dict['current_round_displacement2'][i])
    coefficients = np.array([1, -1.5, -1, -2])
    calculated_reward = sum(coefficients * np.array([block_distance, tower_displacement, tilt, abs(z_rot)]))
    calculated_rewards.append(calculated_reward)
    print(f'[{i}] Block displacement: {block_distance:3.4f}, total_block_distance: {total_block_distance:3.4f}, total tower displacement: {total_tower_displacement:3.4f}, tower_displacement: {tower_displacement:3.4f}, tilt: {tilt:3.4f}, z_rot: {z_rot:3.4f}, reward: {reward:3.4f}, calculated_reward: {calculated_reward:3.4f}')

# add calculated reward to the data
data_dict['calculated_reward'] = calculated_rewards

for key in data_dict:
    minimum = min(data_dict[key])
    maximum = max(data_dict[key])
    mean = np.mean(data_dict[key])
    std = np.std(data_dict[key])
    difference = maximum - minimum
    offset = -minimum
    coefficient = 1/difference
    print(f"{key}: min={minimum:.2f}, max={maximum:.2f}, difference={difference:.2f}, offset={offset:.2f}, coefficient={coefficient:.8f}, mean={mean:.2f}, std={std:.2f}")

# generate code with normalization coefficients
means = []
stds = []
maxes = []
mins = []
for key in data_dict:
    mean = np.mean(data_dict[key])
    std = np.std(data_dict[key])
    maximum = (max(data_dict[key]) - mean) / std
    minimum = (min(data_dict[key]) - mean) / std
    means.append(mean)
    stds.append(std)
    maxes.append(maximum)
    mins.append(minimum)
print('\n')
print(f"state_space_means = np.array({means})")
print(f"state_space_stds = np.array({stds})")
print(f"high = np.array({maxes})")
print(f"low = np.array({mins})")
