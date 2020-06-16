import gym
from stable_baselines.gail import generate_expert_traj
from stable_baselines.common.policies import register_policy, MlpPolicy, FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from jenga import jenga_env_wrapper
import cv2


env = jenga_env_wrapper()

# pre-train using expert trajectories
dataset = ExpertDataset(expert_path='combined_expert_data.npz', verbose=1)

model = PPO2('MlpPolicy', env, verbose=1)
model.pretrain(dataset, n_epochs=500)


obs = env.reset()
reward_sum = 0.0
done = 0
counter = 0
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    reward_sum += reward
    if done:
        cv2.imwrite(f'./last_screenshots/screenshot_{counter}.png', info['last_screenshot'])
        counter += 1
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()

env.close()