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
import logging


log = logging.Logger("file_logger")
file_formatter = logging.Formatter('%(levelname)sMain process:PID:%(process)d:%(funcName)s:%(message)s')
file_handler = logging.FileHandler(filename='pretrain.log', mode='w')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)

def play(model, filename, timeout=900):
    log.debug(f"Start play")
    start_time = time.time()
    seed = time.time_ns()
    try:
        log.debug(f"Start try")
        env = jenga_env_wrapper(normalize=True, seed=seed)
        obs = env.reset()
        pid = env.get_pid()
        log.debug(f"SIM_PID:{pid}:Env reset")
        done = False
        reward_sum = 0
        error = 'All blocks checked!'
        info = {'exception': False, 'toppled': False, 'timeout': True, 'last_screenshot': None, 'extracted_blocks': -1}
        while not done and time.time() - start_time < timeout:
            action, _ = model.predict(obs)
            log.debug(f"SIM_PID:{pid}:For step")
            obs, reward, done, info = env.step(action)
            log.debug(f"SIM_PID:{pid}:After step")
            reward_sum += reward
            if done:
                # cv2.imwrite(f'./last_screenshots/screenshot_{filename}.png', info['last_screenshot'])
                # print(reward_sum)
                log.debug(f"SIM_PID:{pid}:Final result info: {info}")
                reward_sum = 0.0
                log.debug(f"SIM_PID:{pid}:For close")
                obs = env.close()
                log.debug(f"SIM_PID:{pid}:After close")

                if info['toppled']:
                    error = 'Tower toppled!'
                elif info['exception']:
                    error = 'Exception occurred!'
                elif info['timeout']:
                    error = 'Timeout!'

        log.debug(f"SIM_PID:{pid}:Before final close")
        env.close()
        log.debug(f"SIM_PID:{pid}:After final close")

        return SimulationResult(info['extracted_blocks'], time.time() - start_time, 0, error, seed, info['last_screenshot'])
    except:
        log.exception(f"SIM_PID:{pid}:Start except")
        traceback.print_exc()
        return SimulationResult(0, time.time() - start_time, 0, 'Exception in algorithm!', seed, None)



# env = jenga_env_wrapper(normalize=True)
dataset = ExpertDataset(expert_path='/home/bch_svt/cartpole/simulation/rl/expert_data/combined_expert_data_(8)_layer_state_normalized.npz', verbose=1)
runs_per_model = 30
models_num = 6
threads_num = 4

with open('pretrain_results.log', 'a') as f:
    current_time = strftime("%d-%m-%Y_%H:%M:%S", gmtime())
    f.write(f"##########################_{current_time}\n")
    # PID:10244

for epochs in [1000, 900, 800, 700, 600, 500, 400]:
    for i in range(models_num):
        # create environment
        log.debug(f"For create env")
        env = jenga_env_wrapper(normalize=True)
        log.debug(f"After create env")

        # create model and pre-train
        model = PPO2.load('8games_1000epochs')
        # model = PPO2('MlpPolicy', env, verbose=1)
        # model.pretrain(dataset, n_epochs=epochs)
        # model.save(f'./models/model_{epochs}epochs_#{i}')

        results = []
        thread_pool = ThreadPool(threads_num)
        for n in range(runs_per_model):
            res = thread_pool.apply_async(func=play, args=(model, f'{epochs}_{i}_{n}', 600))
            results.append(res)

        while not all(list(map(lambda x: x.ready(), results))):
            log.debug(f"######## Threads status ########")
            for idx in range(len(results)):
                log.debug(f"Thread #{idx}: {'ready' if results[idx].ready() else 'running'}")
            time.sleep(60)

        log.debug(f"Before writing into file")

        for k in range(len(results)):
            with open('pretrain_results.log', 'a') as f:
                f.write(f"{epochs} epochs, model #{i}, run #{k} " + str(results[k].get()) + "\n")
                filename = f'{epochs}_{i}_{k}'
                screenshot = results[k].get().screenshot
                if screenshot is not None:
                    cv2.imwrite(f'./last_screenshots/screenshot_{filename}.png', screenshot)

        log.debug(f"After writing into file")

        average_extracted_blocks = np.mean([e.get().blocks_number for e in results])
        std_extracted_blocks = np.std([e.get().blocks_number for e in results])

        log.debug(f"Before writing2")
        with open('pretrain_results.log', 'a') as f:
            f.write(f"{epochs} epochs, model #{i}: mean: {average_extracted_blocks:.2f}, std: {std_extracted_blocks:.2f}\n")

        log.debug(f"After writing2")