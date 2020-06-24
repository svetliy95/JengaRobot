import gym
from stable_baselines.gail import generate_expert_traj
from stable_baselines.common.policies import register_policy, MlpPolicy, FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from jenga import jenga_env_wrapper
import cv2
import time
from multiprocessing import Pool
import traceback
from multiprocessing.pool import ThreadPool
from jenga import SimulationResult
import numpy as np
from time import gmtime, strftime

def play(model, filename, timeout=900):
    start_time = time.time()
    seed = time.time_ns()
    try:
        env = jenga_env_wrapper(normalize=True, seed=seed)
        obs = env.reset()
        done = False
        reward_sum = 0
        error = 'All blocks checked!'
        info = {'exception': False, 'toppled': False, 'timeout': True, 'last_screenshot': None, 'extracted_blocks': -1}
        while not done and time.time() - start_time < timeout:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                # cv2.imwrite(f'./last_screenshots/screenshot_{filename}.png', info['last_screenshot'])
                print(reward_sum)
                print(f"Final result info: {info}")
                reward_sum = 0.0
                print(f"For close")
                obs = env.close()
                print(f"After close")

                if info['toppled']:
                    error = 'Tower toppled!'
                elif info['exception']:
                    error = 'Exception occurred!'
                elif info['timeout']:
                    error = 'Timeout!'

        return SimulationResult(info['extracted_blocks'], time.time() - start_time, 0, error, seed, info['last_screenshot'])
    except:
        traceback.print_exc()
        return SimulationResult(0, time.time() - start_time, 0, 'Exception in algorithm!', seed, None)



env = jenga_env_wrapper(normalize=True)
dataset = ExpertDataset(expert_path='/home/bch_svt/cartpole/simulation/rl/expert_data/combined_expert_data_(8)_layer_state_normalized.npz', verbose=1)
runs_per_model = 16
models_num = 3
threads_num = 16

with open('pretrain_results.log', 'a') as f:
    current_time = strftime("%d-%m-%Y_%H:%M:%S", gmtime())
    f.write(f"##########################_{current_time}\n")

for epochs in [10, 900, 800, 700, 600, 500, 400]:
    for i in range(models_num):
        # create environment
        env = jenga_env_wrapper(normalize=True)

        # create model and pre-train
        model = PPO2('MlpPolicy', env, verbose=1)
        model.pretrain(dataset, n_epochs=epochs)

        results = []
        thread_pool = ThreadPool(threads_num)
        for n in range(runs_per_model):
            res = thread_pool.apply_async(func=play, args=(model, f'{epochs}_{i}_{n}', 600))
            results.append(res)

        while not all(list(map(lambda x: x.ready(), results))):
            time.sleep(60)

        print(f"Hi!")

        for k in range(len(results)):
            with open('pretrain_results.log', 'a') as f:
                f.write(f"{epochs} epochs, model #{i}, run #{k} " + str(results[k].get()) + "\n")
                filename = f'{epochs}_{i}_{k}'
                cv2.imwrite(f'./last_screenshots/screenshot_{filename}.png', results[k].get().screenshot)

        average_extracted_blocks = np.mean([e.get().blocks_number for e in results])
        std_extracted_blocks = np.std([e.get().blocks_number for e in results])

        with open('pretrain_results.log', 'a') as f:
            f.write(f"{epochs} epochs, model #{i}: mean: {average_extracted_blocks:.2f}, std: {std_extracted_blocks:.2f}\n")