import os
import re
import numpy as np
from time import gmtime, strftime

def get_all_expert_data_files(folder):
    pattern = "expert_data_30_"
    all_filenames = os.listdir(folder)
    filenames = []
    for filename in all_filenames:
        if bool(re.search(pattern, filename)):
            filenames.append(folder + filename)
    return filenames

def open_expert_data(filenames):
    expert_data = {'actions': np.reshape(np.array([]), (0, 1)),
                   'obs': np.reshape(np.array([]), (0, 15)),
                   'rewards': np.array([]),
                   'episode_returns': np.array([]),
                   'episode_starts': np.array([])}

    for filename in filenames:
        data = np.load(filename)
        for key in expert_data.keys():
            print(expert_data[key].shape)
            print(data[key].shape)
            expert_data[key] = np.concatenate((expert_data[key], data[key]), 0)

    return expert_data

def combine_expert_data(filenames, save_as_file: bool):
    expert_data = open_expert_data(filenames)

    if save_as_file:
        current_time = strftime("%d-%m-%Y_%H:%M:%S", gmtime())
        np.savez(f'combined_expert_data_{current_time}', **expert_data)

    return expert_data

def calculate_normalization_coefficients(filenames):
    expert_data = combine_expert_data(filenames, False)
    obs = expert_data['obs']
    rewards = expert_data['rewards']
    actions = expert_data['actions']
    print(obs)

    state_space_means = obs.mean(axis=0)
    state_space_stds = obs.std(axis=0)
    reward_mean = rewards.mean(axis=0)
    reward_std = rewards.std(axis=0)
    action_mean = actions.mean()
    action_std = actions.std()

    print(f"state_space_means = np.array({list(state_space_means)})")
    print(f"state_space_stds = np.array({list(state_space_stds)})")
    print(f"reward_mean = {reward_mean}")
    print(f"reward_std = {reward_std}")
    print(f"action_mean = {action_mean}")
    print(f"action_std = {action_std}")

    return state_space_means, state_space_stds, reward_mean, reward_std

def normalize_expert_data(filename, save_as_file):
    expert_data = open_expert_data(filenames=[filename])
    state_space_means, state_space_stds, reward_mean, reward_std = calculate_normalization_coefficients([filename])
    expert_data['obs'] -= state_space_means
    expert_data['obs'] /= state_space_stds
    expert_data['rewards'] -= reward_mean
    expert_data['rewards'] /= reward_std

    if save_as_file:
        filename_without_extension = os.path.splitext(filename)[0]
        current_time = strftime("%d-%m-%Y_%H:%M:%S", gmtime())
        np.savez(f'{filename_without_extension}_normalized', **expert_data)

    return expert_data

if __name__ == "__main__":
    filename = 'expert_data/expert_data_30blocks_20extracted.npz'
    filenames = get_all_expert_data_files('./expert_data/')
    print(filenames)
    # calculate_normalization_coefficients(filenames)
    #
    # combine_expert_data(filenames, True)

    normalize_expert_data('/home/bch_svt/cartpole/simulation/rl/expert_data/combined_expert_data_(8)_layer_state.npz', True)
