import gym
from stable_baselines.gail import generate_expert_traj
from stable_baselines.common.policies import register_policy, MlpPolicy, FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from cartpole_custom import CartPoleCustomEnv


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[64, dict(pi=[64],
                                                          vf=[64])],
                                           feature_extraction="mlp")

# Register the policy, it will check that the name is not already taken
register_policy('CustomPolicy', CustomPolicy)


# generate expert trajectories
# env = gym.make('CartPole-v1')
env = CartPoleCustomEnv(0)
model = PPO2('CustomPolicy', env, verbose=1, tensorboard_log='./tb_log', learning_rate=0.0020)
generate_expert_traj(model, 'expert_cartpole', n_timesteps=10**5, n_episodes=10, n)

# pre-train using expert trajectories
dataset = ExpertDataset(expert_path='expert_cartpole.npz')
model2 = PPO2('MlpPolicy', env, verbose=1)
model2.pretrain(dataset, n_epochs=100)


obs = env.reset()
reward_sum = 0.0
for _ in range(1000):
        action, _ = model2.predict(obs)
        obs, reward, done, _ = env.step(action)
        reward_sum += reward
        env.render()
        if done:
                print(reward_sum)
                reward_sum = 0.0
                obs = env.reset()

env.close()