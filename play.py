from stable_baselines import PPO2
from real_jenga_env import jenga_env

# create gym environment
env = jenga_env(normalize=True)

# load pretrained model
model = PPO2.load('./models/real_robot/pushing/real_model_2000epochs_7features_seems_to_be_good.zip')

# reset gym env
done = False
reward_sum = 0
obs = env.reset()

while not done:
    # get an action based on observation (system state)
    action, _ = model.predict(obs)

    # perform the action
    obs, reward, done, info = env.step(action)
    reward_sum += reward


