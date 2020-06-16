from jenga import jenga_env_wrapper
from pynput import keyboard
import time
import csv
import numpy as np


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

def wait_for_expert_action():
    while action is None:
        time.sleep(0.1)


action = None
start_keyboard_listener()
env = jenga_env_wrapper()
obs = env.reset()

dataset = []
reward = 0
done = False

filename = f'observation_data{time.time_ns()}.csv'

for i in range(10):
    wait_for_expert_action()
    # dataset.append(list(obs) + [action] + [reward] + [1 if done else 0])
    datarow = list(obs) + [action] + [reward] + [1 if done else 0]

    with open(filename, mode='a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(datarow)
    obs, reward, done, info = env.step(action)
    action = None  # reset action

    if info['exception']:
        obs = env.reset()
    if done:
        obs = env.reset()
    print(f"Sample #{i}")

env.close()

