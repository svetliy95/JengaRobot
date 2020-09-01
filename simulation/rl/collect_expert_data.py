from stable_baselines.gail import generate_expert_traj
from stable_baselines.common.policies import register_policy, MlpPolicy, FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from jenga import jenga_env_wrapper
from time import gmtime, strftime
from pynput import keyboard
import os
import time
from play import jenga_env

def start_keyboard_listener():
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

def on_press(key):
    global action
    try:
        if key.char == '-':
            action = 0
        if key.char == '+':
            action = 1
    except AttributeError:
        pass

def expert_input(obs):
    global action
    action = None
    while action is None:
        time.sleep(0.1)
    return action


action = None

# create environment
# env = jenga_env(normalize=False)
env = jenga_env_wrapper(normalize=False)

# start keyboard listener
start_keyboard_listener()

current_time = strftime("%d-%m-%Y_%H:%M:%S", gmtime())
generate_expert_traj(expert_input, f'expert_data{current_time}', env, n_episodes=1)
