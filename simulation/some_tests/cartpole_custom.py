import gym
import numpy as np
import math
from gym.envs.classic_control.cartpole import CartPoleEnv

class CartPoleCustomEnv(CartPoleEnv):
    def __init__(self, reward_function_id):
        super().__init__()
        self.coefficients = np.array([1000, 1000, 1000, 1000])
        self.coefficients = np.array([1, 1, 1, 1])
        self.reward_function_id = reward_function_id
        # bounds_coefficients = np.array([self.coefficients[0], 1, self.coefficients[2], 1])
        # self.observation_space.high *= bounds_coefficients
        # self.observation_space.low *= bounds_coefficients

    def step(self, action):
        state, reward, done, info = super().step(action)

        normalized_distance = abs(state[0]) / 2.4
        normalized_distance = normalized_distance

        normalized_angle = abs(state[2])/41.8
        normalized_angle = normalized_angle

        velocity = abs(state[1])

        # penalize distance
        if self.reward_function_id == 1:

            new_reward = reward - normalized_distance
            return state, new_reward, done, info

        # additionally penalize angle
        elif self.reward_function_id == 2:
            new_reward = reward - normalized_distance - normalized_angle
            return state, new_reward, done, info

        # additionally penalize velocity
        elif self.reward_function_id == 3:
            # new_reward = reward - normalized_distance**2 - (5*normalized_angle)**2 #- velocity**2
            new_reward = reward - (10*normalized_angle)**2 - (10*normalized_distance)**2  #- velocity**2
            new_reward = reward * (1/((20*normalized_angle))) * (1/(25*normalized_distance))
            new_reward = reward - (100*normalized_angle) - (10*normalized_distance)
            new_reward = reward - (50*normalized_angle) - 2*normalized_distance
            return state, new_reward, done, info

        elif self.reward_function_id == 4:
            # print(f"State before: {state}")
            new_state = state * self.coefficients
            # print(f"State after: {new_state}")

            return new_state, reward, done, info

        else:
            return state, reward, done, info