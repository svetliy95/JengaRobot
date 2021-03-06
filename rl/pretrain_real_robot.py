import gym
from stable_baselines.gail import generate_expert_traj
from stable_baselines.common.policies import register_policy, MlpPolicy, FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from real_jenga_env import jenga_env

# create gym environment
env = jenga_env(normalize=True)

# load collected training data
# dataset = ExpertDataset(expert_path='/home/bch_svt/cartpole/simulation/rl/training_data/real/without_ids/combined_expert_data_29-08-2020_11:08:26_normalized.npz', verbose=1, train_fraction=0.8)
dataset = ExpertDataset(expert_path='/rl/training_data/real/without_ids_&_without_configuration/combined_expert_data_without_ids_&_configuration_normalized.npz', verbose=1, train_fraction=0.8)

# choose epochs number
epochs = 2000


# uncomment this rows if you want to retrain the model
# model = PPO2('MlpPolicy', env, verbose=1)
# model.pretrain(dataset, n_epochs=epochs)
# model.save(f'./models/real_model_{epochs}epochs')

# load pretrained model
model = PPO2.load('/rl/models/real_model_2000epochs_7features_seems_to_be_good.zip')

# reset gym env
done = False
reward_sum = 0
obs = env.reset()

while not done:
    print(f"Before predict!")
    # get an action based on observation (system state)
    action, _ = model.predict(obs)
    print(f"After predict!")

    # perform the action
    obs, reward, done, info = env.step(action)
    print(f"Done: {done}")
    print(f"Info: {info}")
    print(f"After step!")
    reward_sum += reward


