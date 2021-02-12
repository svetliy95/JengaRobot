import os
import re
import numpy as np
from time import gmtime, strftime
import glob
import pandas as pd

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
                   'obs': np.reshape(np.array([]), (0, 7)),
                   'rewards': np.array([]),
                   'episode_returns': np.array([]),
                   'episode_starts': np.array([])}

    for filename in filenames:
        data = np.load(filename)
        for key in expert_data.keys():
            print(f"Key: {key}")
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
    high = obs.max(axis=0)
    low = obs.min(axis=0)

    print(f"state_space_means = np.array({list(state_space_means)})")
    print(f"state_space_stds = np.array({list(state_space_stds)})")
    print(f"reward_mean = {reward_mean}")
    print(f"reward_std = {reward_std}")
    print(f"action_mean = {action_mean}")
    print(f"action_std = {action_std}")
    print(f"high = np.{repr(high)}")
    print(f"low = np.{repr(low)}")

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

    # folder = '/home/bch_svt/cartpole/simulation/rl/training_data/real/raw/*with_ids*'
    # fnames = glob.glob(folder)
    #
    # combined_data_with_ids = combine_expert_data(fnames, False)
    #
    # # remove ids from data
    # combined_data_with_ids['obs'] = combined_data_with_ids['obs'][:, :-1]
    #
    # np.savez(f'combined_expert_data_without_ids_part1', **combined_data_with_ids)

    # folder = '/home/bch_svt/cartpole/simulation/rl/training_data/real/raw/*10_blocks*'
    # fnames = glob.glob(folder)
    #
    # combined_data_with_ids = combine_expert_data(fnames, False)
    #
    # np.savez(f'combined_expert_data_without_ids_part2', **combined_data_with_ids)
    #
    # fnames = ['combined_expert_data_without_ids_part1.npz', 'combined_expert_data_without_ids_part2.npz']
    #
    # combine_expert_data(fnames, True)

    # fnames = ['/home/bch_svt/cartpole/simulation/rl/training_data/real/without_ids/combined_expert_data_29-08-2020_11:08:26.npz']
    #
    # training_data = open_expert_data(fnames)
    #
    # training_data['obs'] = training_data['obs'][:, :-2]
    #
    # print(f"Expert data obs: {training_data['obs'][0]}")
    #
    # def find_such_row(obs, actions):
    #     for i in range(len(obs)):
    #         ob1 = obs[i][0]
    #         ob2 = obs[i][1]
    #         ob3 = obs[i][2]
    #         ob4 = obs[i][3]
    #         ob5 = obs[i][4]
    #         ob6 = obs[i][5]
    #         if ob1 == 0 and ob2 == 0 and ob3 == 0 and ob4 == 0 and ob5 == 0 and ob6 == 0 and actions[i][0] == 0:
    #             return i
    #
    #     return -1
    #
    # print(f"Keys: {training_data.keys()}")
    #
    # episode_rewards = []
    # j = 0
    # for i in range(len(training_data['obs'])):
    #     if training_data['episode_starts'][0] == 1:
    #         episode_rewards.append(training_data['episode_returns'][j])
    #     else:
    #         episode_rewards.append(0.0)
    #
    # while find_such_row(training_data['obs'], training_data['actions']) != -1:
    #     i = find_such_row(training_data['obs'], training_data['actions'])
    #     training_data['obs'] = np.delete(training_data['obs'], i, 0)
    #     training_data['actions'] = np.delete(training_data['actions'], i, 0)
    #     training_data['rewards'] = np.delete(training_data['rewards'], i, 0)
    #     print(f"Episode starts: {training_data['episode_starts'][i]}")
    #     if training_data['episode_starts'][i] == 1:
    #         print(f"Episode starts i: {i}")
    #         print(f"Reward: {episode_rewards[i]}")
    #     training_data['episode_starts'] = np.delete(training_data['episode_starts'], i, 0)
    #
    # print(f"{find_such_row(training_data['obs'], training_data['actions'])}")
    # print(f"{training_data['episode_returns']}")
    #
    # training_data['episode_returns'] = np.delete(training_data['episode_returns'], 0, 0)
    #
    # np.savez(f'combined_expert_data_without_ids_&_configuration', **training_data)


    # fnames = ['/home/bch_svt/cartpole/simulation/rl/training_data/real/without_ids_&_without_configuration/combined_expert_data_without_ids_&_configuration.npz']
    #
    # training_data = open_expert_data(fnames)
    #
    # training_data['obs'] = training_data['obs'][:, :-1]
    #
    # print(f"Expert data obs: {training_data['obs'].shape}")
    #
    # np.savez(f'combined_expert_data_without_ids_&_configuration_&_heights', **training_data)

    fnames = ['/home/bch_svt/cartpole/simulation/rl/training_data/real/without_ids_&_without_configuration/combined_expert_data_without_ids_&_configuration.npz']
    calculate_normalization_coefficients(fnames)
    normalize_expert_data(
        '/rl/training_data/real/without_ids_&_without_configuration/combined_expert_data_without_ids_&_configuration.npz', True)






    exit()

    fname = '/home/bch_svt/cartpole/simulation/rl/expert_data21-08-2020_13:32:00.npz'
    calculate_normalization_coefficients([fname])

    exit()

    filename = '../training_data/expert_data_30blocks_20extracted.npz'
    filenames = get_all_expert_data_files('./training_data/')
    print(filenames)
    # calculate_normalization_coefficients(filenames)
    #
    # combine_expert_data(filenames, True)

    normalize_expert_data('/rl/training_data/combined_expert_data_(8)_layer_state.npz', True)
