import gym

from stable_baselines.common.policies import MlpPolicy, ActorCriticPolicy, FeedForwardPolicy, MlpLnLstmPolicy, register_policy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from cartpole_custom import CartPoleCustomEnv
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines.gail import generate_expert_traj


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[64, dict(pi=[64],
                                                          vf=[64])],
                                           feature_extraction="mlp")


# Register the policy, it will check that the name is not already taken
register_policy('CustomPolicy', CustomPolicy)



# multiprocess environment
# env = make_vec_env('CartPole-v1', n_envs=2)
# env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
# env = VecNormalize(env, norm_obs=True, norm_reward=True)


# env = gym.make('CartPole-v1')

env = DummyVecEnv([lambda: CartPoleCustomEnv(0)])
# env = CartPoleCustomEnv(0)
env = VecNormalize(env, norm_obs=True, norm_reward=True)


# env = CartPoleCustomEnv(3)





model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log='./tb_log')
model.learn(total_timesteps=50000)
model.save("ppo2_cartpole")

del model  # remove to demonstrate saving and loading

model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
obs = env.reset()

N = 100
durations = []

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


for i in range(N):
    timestep = 0
    fl = True
    dones = False
    while not dones:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(f"Original reward: {env.get_original_reward()}, new reward: {rewards}")
        print(f"Original obs: {env.get_original_obs()}, new obs: {obs}")
        print()
        env.render(mode='human')
        if dones and fl:
            print(timestep)
            fl = False

        timestep += 1

    env.reset()
    durations.append(timestep)

print(f"Mean: {np.mean(durations)}")
print(f"Std: {np.std(durations)}")
print(f"Median: {np.median(durations)}")
#
# x = np.linspace(0, N, N)
# plt.plot(x, durations)
# plt.show()

# vanilla
# mean = 528
# median = 456


