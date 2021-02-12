import gym
from stable_baselines.gail import generate_expert_traj
from stable_baselines.common.policies import register_policy, MlpPolicy, FeedForwardPolicy
from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset
from simulation.jenga import jenga_env_wrapper
import cv2
import time
from multiprocessing import Pool
import traceback
from multiprocessing.pool import ThreadPool
from simulation.jenga import SimulationResult
import numpy as np
from time import gmtime, strftime
import logging
import os
import json
from real_jenga_env import jenga_env


log = logging.Logger(__name__)
file_formatter = logging.Formatter('%(asctime)s:%(levelname)sMain process:PID:%(process)d:%(funcName)s:%(message)s')
file_handler = logging.FileHandler(filename='pretrain.log', mode='w')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)

def play(model, filename, timeout=900, seed=None):
    log.debug(f"Start play")
    start_time = time.time()
    if seed is None:
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
                log.debug(f"Reward sum: {reward_sum}")
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


def write_results_json(epochs, model_id, run, result, filename):
    # this function should preserve data stored in file
    # for this purpose we first read all data from the file

    # create file if not exists
    if not os.path.exists(filename):
        with open(filename, mode='w') as f:
            d = []
            json.dump(d, f)

    with open(filename, mode='r') as f:
        data = json.load(f)

    # add new data row to the data
    data.append({'epochs': epochs, 'model_id': model_id, 'run': run, 'blocks_number': result.blocks_number,
                 'real_time': result.real_time, 'sim_time': result.sim_time,
             'error': result.error, 'seed': result.seed, 'real_time_factor': result.real_time_factor})
    # data[str(tag_id)] = {'pos': measured_pos, 'quat': measured_quat, 'pos_std': pos_std, 'quat_std': quat_std}

    print(data)

    # write data back to the file
    with open(filename, mode='w') as f:
        json.dump(data, f)
    print(f"Data for the model #{model_id}({epochs}/{run}) successfully writen!")



# env = jenga_env_wrapper(normalize=True)
dataset = ExpertDataset(expert_path='../training_data/simulation/combined_expert_data_(8)_layer_state_normalized.npz', verbose=1)
runs_per_model = 1
models_num = 1
threads_num = 1
timeout = 900
seeds = [16041995, 14021996]

with open('pretrain_results.log', 'a') as f:
    current_time = strftime("%d-%m-%Y_%H:%M:%S", gmtime())
    f.write(f"##########################_{current_time}\n")
    # PID:10244

# for epochs in [1000, 900, 800, 700, 600, 500, 400]:
#     for i in range(models_num):
#         # create environment
#         log.debug(f"For create env")
#         env = jenga_env_wrapper(normalize=True)
#         log.debug(f"After create env")
#
#         # create model and pre-train
#         # model = PPO2.load('8games_1000epochs')
#         model = PPO2('MlpPolicy', env, verbose=1)
#         model.pretrain(dataset, n_epochs=epochs)
#         model.save(f'./models/model_{epochs}epochs_#{i}')
#
#         results = []
#         thread_pool = ThreadPool(threads_num)
#         for n in range(runs_per_model):
#             res = thread_pool.apply_async(func=play, args=(model, f'{epochs}_{i}_{n}', 600))
#             results.append(res)
#
#         while not all(list(map(lambda x: x.ready(), results))):
#             log.debug(f"######## Threads status ########")
#             for idx in range(len(results)):
#                 log.debug(f"Thread #{idx}: {'ready' if results[idx].ready() else 'running'}")
#             time.sleep(60)
#
#         log.debug(f"Before writing into file")
#
#         for k in range(len(results)):
#             with open('pretrain_results.log', 'a') as f:
#                 f.write(f"{epochs} epochs, model #{i}, run #{k} " + str(results[k].get()) + "\n")
#                 filename = f'{epochs}_{i}_{k}'
#                 screenshot = results[k].get().screenshot
#                 if screenshot is not None:
#                     cv2.imwrite(f'./last_screenshots/screenshot_{filename}.png', screenshot)
#
#         log.debug(f"After writing into file")
#
#         average_extracted_blocks = np.mean([e.get().blocks_number for e in results])
#         std_extracted_blocks = np.std([e.get().blocks_number for e in results])
#
#         log.debug(f"Before writing2")
#         with open('pretrain_results.log', 'a') as f:
#             f.write(f"{epochs} epochs, model #{i}: mean: {average_extracted_blocks:.2f}, std: {std_extracted_blocks:.2f}\n")
#
#         log.debug(f"After writing2")


for epochs in [1000, 900, 800, 700, 600, 500, 400]:
    # create environment
    log.debug(f"For create env")
    env = jenga_env_wrapper(normalize=True)
    log.debug(f"After create env")
    models = []
    for i in range(models_num):
        # model = PPO2('MlpPolicy', env, verbose=1)
        # model.pretrain(dataset, n_epochs=epochs)
        # model.save(f'./models/model_{epochs}epochs_#{i}')
        model = PPO2.load('../models/simulation/pushing/model_900epochs_#3.zip')
        models.append(model)
    for m in range(len(models)):
        results = []
        thread_pool = ThreadPool(threads_num)
        for seed in seeds:
            for i in range(runs_per_model):
                res = thread_pool.apply_async(func=play, args=(models[m], f'{epochs}_{m}_{i}', timeout, seed))
                results.append(res)

        # create model and pre-train
        # model = PPO2.load('8games_1000epochs')while not all(li

        while not all(list(map(lambda x: x.ready(), results))):
            log.debug(f"######## Threads status ########")
            for idx in range(len(results)):
                log.debug(f"Thread #{idx}: {'ready' if results[idx].ready() else 'running'}")
            time.sleep(60)

        log.debug(f"Before writing into file")

        for k in range(len(results)):
            # write to json
            write_results_json(epochs, m, k, results[k].get(), 'pretrain_results_full_run.json')

            with open('pretrain_results.log', 'a') as f:
                f.write(f"{epochs} epochs, model #{m}, run #{k} " + str(results[k].get()) + "\n")
                filename = f'{epochs}_{m}_{k}'
                screenshot = results[k].get().screenshot
                if screenshot is not None:
                    cv2.imwrite(f'./last_screenshots/screenshot_{filename}.png', screenshot)

        log.debug(f"After writing into file")

        average_extracted_blocks = np.mean([e.get().blocks_number for e in results])
        std_extracted_blocks = np.std([e.get().blocks_number for e in results])

        log.debug(f"Before writing2")
        with open('pretrain_results.log', 'a') as f:
            f.write(f"{epochs} epochs, model #{m}: mean: {average_extracted_blocks:.2f}, std: {std_extracted_blocks:.2f}\n")

        log.debug(f"After writing2")