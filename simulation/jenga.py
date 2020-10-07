from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen
from pynput import keyboard
import os
from generate_scene import generate_scene
import numpy as np
from pyquaternion import Quaternion
import time
from threading import Thread, Lock

# animated plot
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.animation as animation
style.use('fivethirtyeight')
from constants import *
from pusher import Pusher
from tower import Tower
from extractor import Extractor
from math import sin, cos, radians
import cv2
import math
import logging
import colorlog
import gym
import collections
from multiprocessing import Process, Queue, Pool
from enum import Enum, auto
import cProfile
from pyinstrument import Profiler
import os
import traceback
import glfw
import sys
from utils.utils import get_angle_between_quaternions_3ax
import csv
import random
import multiprocessing_logging
import copy

profiler = Profiler()

# specify logger
# DEBUG: Detailed information, typically of interest only when diagnosing problems.
# INFO: Confirmation that things are working as expected.
# WARNING: An indication that something unexpected happened, or indicative of some problem in
# the near future (e.g. ‘disk space low’). The software is still working as expected.
# ERROR: Due to a more serious problem, the software has not been able to perform some function.
# CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
log = logging.Logger(__name__)
# formatter = colorlog.ColoredFormatter('%(log_color)s%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)sPID:%(process)d:%(funcName)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
log.addHandler(stream_handler)

log = logging.Logger(__name__)
file_formatter = logging.Formatter('%(asctime)s:%(levelname)s:PID:%(process)d:%(funcName)s:%(message)s')
file_handler = logging.FileHandler(filename='jenga.log', mode='w')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)

# important line, without it there is a concurrency problem between processes
multiprocessing_logging.install_mp_handler(log)


class SimulationResult:
    def __init__(self, blocks_num, real_time, sim_time, error, seed, screenshot, extracted_blocks=[]):
        self.extracted_blocks = extracted_blocks
        self.blocks_number = blocks_num
        self.real_time = real_time
        self.sim_time = sim_time
        self.error = error
        self.seed = seed
        self.real_time_factor = sim_time / real_time
        self.screenshot = screenshot

    def __str__(self):
        return f"Termination reason: {self.error}. " + \
        f"Blocks number: {self.blocks_number}. """ + \
        f"Elapsed time: {self.real_time:.0f}s. " + \
        f"Real-time factor: {self.real_time_factor:.2f}. " + \
        f"Seed: {self.seed}. " + \
        f"Extracted blocks: {self.extracted_blocks}."


class Error(Enum):
    tower_toppled = auto()
    exception_occurred = auto()
    timeout = auto()


class Action(Enum):
    push = auto()
    pull_and_place = auto()
    move_to_block = auto()

# class Action:
#     def __init__(self, type: ActionType, lvl, pos):
#         assert type in [ActionType.push, ActionType.pull_and_place, ActionType.move_to_block], "Wrong action!"
#         assert lvl in range(15), "Wrong level index!"
#         assert pos in range(3), "Wrong position index!"
#         self.type = type
#         self.lvl = lvl
#         self.pos = pos


class Status:
    ready = 'ready'
    ready_to_push = 'ready_to_push'
    pushing = 'pushing'
    over = 'over'


class State:
    def __init__(self,
                 tower_state,
                 force,
                 block_distance,
                 total_block_distance,
                 tilt,
                 blocks_displacement_total,
                 blocks_displacement_last,
                 z_rotation_total,
                 z_rotation_last,
                 status,
                 current_block_height,
                 side
                 ):
        self.tower_state = tower_state
        self.force = force
        self.block_distance = block_distance
        self.total_block_distance = total_block_distance
        self.tilt = tilt
        self.blocks_displacement_total = blocks_displacement_total
        self.blocks_displacement_last = blocks_displacement_last
        self.z_rotation_total = z_rotation_total
        self.z_rotation_last = z_rotation_last
        self.status = status
        self.current_block_height = current_block_height
        self.side = side

    def __str__(self):
        return f"""
        Tower state: {str(self.tower_state)}
        Force: {str(self.force)}
        Block distance: {str(self.block_distance)}
        Total block distance: {str(self.total_block_distance)}
        Tilt: {str(self.tilt)}
        Block displacement total: {str(self.blocks_displacement_total)}
        Block displacement last: {str(self.blocks_displacement_last)}
        Z rotation total: {str(self.z_rotation_total)}
        Z rotation last: {str(self.z_rotation_last)}
        Status: {str(self.status)}
        Current block height: {str(self.current_block_height)}
        Side: {str(self.side)}"""


class jenga_env(gym.Env):

    # action_space = gym.spaces.Discrete(3)
    # observation_space = gym.spaces.Box(1)


    def __init__(self, render=True, timeout=600, seed=None):
        self.exception_fl = False
        self.toppled_fl = False
        self.timeout_fl = False
        self.screenshot_fl = False
        self.abort_fl = False
        self.placing_fl = False

        log.debug(f"Process #{os.getpid()} started with seed: {seed}")

        # debugging stuff
        self.localization_errors = []

        # global flags
        self.render = render
        self.on_screen_rendering = False
        self.plot_force = False
        self.key_board_listener_fl = False
        self.screenshot_fl = False
        self.tower_toppled_fl = False
        self.simulation_aborted = False
        self.simulation_over = False
        self.plot_env_state = True

        # globals
        self.last_screenshot = None
        self.last_state = None
        self.last_reward = None
        self.previous_step_displacement = None
        self.previous_step_z_rot = None
        self.extracted_blocks = 0


        # sensor data
        self.g_sensor_data_queue = []
        self.g_sensors_data_queue_maxsize = 250
        self.g_state_data_queue = []
        self.g_state_data_queue_maxsize = 250

        # initialize simulation
        scene_xml, self.seed = generate_scene(g_blocks_num, g_timestep, seed=seed)
        self.model = load_model_from_xml(scene_xml)
        self.sim = MjSim(self.model)
        self.pause_fl = False
        self.viewer = None
        self.t = 0
        # blocks_num - 9 + 3 is the maximum height of a tower with blocks num blocks
        self.checked_positions = {i: {j for j in range(3)} for i in range(g_blocks_num - 9 + 3)}

        # declare internal entities
        self.tower = None
        self.pusher = None
        self.extractor = None

        # initialize state variables
        self.total_distance = 0
        self.current_block_id = 0
        self.current_lvl = 0
        self.current_lvl_pos = 0
        self.initial_tilt_2ax = np.array([0, 0])
        self.previous_tilt_1ax = 0
        self.steps_pushed = 0
        self.max_pushing_distance = 25

        # # create histogram plot for position errors
        # self.x_errors = np.arange(g_blocks_num)
        # self.y_errors = np.zeros(g_blocks_num)
        # plt.ion()
        # self.fig_errors = plt.figure()
        # self.ax = self.fig_errors.add_subplot(111)
        # self.ax.set(ylim=(0, 15))
        # self.ax.xaxis.set_ticks(range(g_blocks_num))
        # self.ax.yaxis.set_ticks(range(10))
        # self.line1 = self.ax.bar(self.x_errors, self.y_errors)
        # self.ax.grid(zorder=0)

        # start keyboard listener
        if self.key_board_listener_fl:
            self.start_keyboard_listener()

        # start plotting process
        if self.plot_force:
            self.force_data_q = Queue()
            plotting_process = Process(target=self.plot_force_data, args=(self.force_data_q,))
            plotting_process.start()

        if self.plot_env_state:
            self.env_state_data_q = Queue()
            state_plotting_process = Process(target=self.plot_env_state_data, args=(self.env_state_data_q,))
            state_plotting_process.start()


        # start force data plotting
        # if self.plot_force:
        #     plotting_thread = Thread(target=self.plot_force_data, daemon=True)
        #     plotting_thread.start()

        # start simulation
        log.debug(f"Start simulation thread")
        self.simulation_thread = Thread(target=self.simulate, daemon=True)
        self.simulation_thread.start()

        log.debug(f"Simulation thread started!")

        # wait until pusher and extractor are initialized
        while self.pusher is None or self.extractor is None or self.tower is None:
            time.sleep(0.1)

        self.positions_from_last_step = copy.deepcopy(self.tower.ref_positions)

        log.debug(f"Pusher and extractor are ready!")

        # move pusher to the starting position
        self.move_to_random_block()

        log.debug(f"Env initialization done!")

    def off_screen_render_and_plot_errors(self, line1, fig_errors):
        positions, im1, im2 = self.tower.get_poses_cv(range(g_blocks_num))

        # get actual and estimated positions
        actual_positions = []
        for i in range(g_blocks_num):
            actual_positions.append(self.tower.get_position(i))
        actual_positions = np.array(actual_positions)

        # extract estimated positions
        h, w, _ = im1.shape
        new_h = int(1080 / 2.2)
        new_w = int(1920 / 2.2)
        im1 = cv2.resize(im1, (new_w, new_h))
        im2 = cv2.resize(im2, (new_w, new_h))
        black_line = np.zeros((new_h, 10, 3), np.uint8)
        im = np.concatenate((im1, black_line, im2), axis=1)
        cv2.imshow(mat=im, winname="Render")
        start = time.time()
        cv2.waitKey(1)
        stop = time.time()
        # print(f"Wait_time: {stop - start} s.")
        for i in range(g_blocks_num):
            est_pose = positions.get(i)
            if est_pose is not None:
                pos_error = est_pose['orientation_error']
                line1[i].set_height(pos_error)
                if est_pose['tags_detected'] == 2:
                    line1[i].set_color('b')
                else:
                    line1[i].set_color('y')
            else:
                line1[i].set_height(10)
                line1[i].set_color('r')
        fig_errors.canvas.draw()
        fig_errors.canvas.flush_events()

    def print_fixed_camera_xml(cam_pos, cam_lookat):
        cam_zaxis = cam_pos - cam_lookat
        plane_normal = np.array([0, 0, 1])
        cam_xaxis = -np.cross(cam_zaxis, plane_normal)
        cam_yaxis = np.cross(cam_zaxis, cam_xaxis)

        print(f'pos="{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}"'
              f' xyaxes="{cam_xaxis[0]} {cam_xaxis[1]} {cam_xaxis[2]} {cam_yaxis[0]} {cam_yaxis[1]} {cam_yaxis[2]}"')

    def get_camera_pose(self):
        # TODO: return orientation
        elevation = radians(-self.viewer.cam.elevation)
        azimuth = radians(self.viewer.cam.azimuth)
        lookat = np.array(self.viewer.cam.lookat)
        distance = self.viewer.cam.distance

        z = lookat[2] + sin(elevation) * distance
        proj_dist = cos(elevation) * distance

        x = lookat[0] - proj_dist * cos(azimuth)
        y = lookat[1] - proj_dist * sin(azimuth)

        print(f"Camera elevation: {self.viewer.cam.elevation}, azimuth: {self.viewer.cam.azimuth}, lookat: {self.viewer.cam.lookat}")

        return np.array([x, y, z, 0, 0, 0])

    def plot_force_data(self, force_data_q):
        print(f"Hi from process!")
        def animate_force_data(i):
            ys = force_data_q.get()
            print(f"ys: {ys}")
            ys = -np.array(ys)
            xs = range(0, len(ys))
            ax1_force.clear()
            ax1_force.plot(xs, ys)

        # define plot for the force data plotting
        force_figure = plt.figure()
        ax1_force = force_figure.add_subplot(1, 1, 1)
        ani = animation.FuncAnimation(force_figure, animate_force_data, interval=1)
        plt.show()

        # fig = plt.figure()
        # canvas = np.zeros((48, 64))
        # screen = pf.screen(canvas, 'Force values')
        #
        # while True:
        #     fig.clear()
        #     ys = -np.array(self.g_sensor_data_queue)  # copy data because of no thread synchronisation
        #     xs = range(-len(ys) + 1, 1)
        #     plt.plot(xs, ys, c='black')
        #     fig.canvas.draw()
        #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        #     screen.update(image)
        #     time.sleep(0.001)

    def update_force_sensor_plot(self):
        current_sensor_value = self.sim.data.sensordata[g_blocks_num*3 + g_blocks_num*4]
        if len(self.g_sensor_data_queue) >= self.g_sensors_data_queue_maxsize:
            self.g_sensor_data_queue.pop(0)
        self.g_sensor_data_queue.append(current_sensor_value)
        if not self.force_data_q.empty():
            # self.force_data_q.get()
            self.force_data_q.put(self.g_sensor_data_queue)
        else:
            self.force_data_q.put(self.g_sensor_data_queue)

    def plot_env_state_data(self, env_state_data_q):
        def animate_state_data(i):
            if not env_state_data_q.empty():
                data = env_state_data_q.get()
            else:
                return

            states = [i[0] for i in data]
            rewards = [i[1] for i in data]

            forces = [i[0] for i in states]
            block_displacements = [i[1] for i in states]
            total_displacements = [i[2] for i in states]
            current_step_tower_disp = [i[3] for i in states]
            current_round_tower_disp = [i[4] for i in states]
            current_step_max_disp = [i[5] for i in states]
            block_height = [i[6] for i in states]

            ys = []
            ys.append(current_step_tower_disp)
            ys.append(current_step_max_disp)
            ys.append(block_displacements)
            ys.append(forces)
            ys.append(rewards)

            axes_titles = ['Current step average displacement', 'Current step max displacement', 'Block displacement', 'Force', 'Reward']


            y_limits = []
            y_limits.append([-1, 3])
            y_limits.append([-1, 3])
            y_limits.append([-1, 3])
            y_limits.append([-0.3, 0.7])
            y_limits.append([-5, 5])
            y_limits.append([-0.5, 0.5])

            # x-data
            xs = range(0, len(forces))

            # update plots

            for i in range(n):
                axs[i].clear()
                axs[i].set_ylim(y_limits[i])
                axs[i].title.set_text(axes_titles[i])
                axs[i].plot(xs, ys[i])

        # define plot for the force data plotting
        n = 5
        state_figure, axs = plt.subplots(nrows=3, ncols=2)
        axs = np.reshape(axs, (6,))

        # state_figure = plt.figure()
        # axes = []
        # for i in range(1, rows_n + 1):
        #     ax = state_figure.add_subplot(rows_n, 2, i)
        #     axes.append(ax)
        state_figure.tight_layout()

        ani = animation.FuncAnimation(state_figure, animate_state_data, interval=1)
        plt.show()

    def update_env_state_plot(self, state, reward):
        if len(self.g_state_data_queue) >= self.g_state_data_queue_maxsize:
            self.g_state_data_queue.pop(0)
        self.g_state_data_queue.append((state, reward))
        if not self.env_state_data_q.empty():
            # self.force_data_q.get()
            self.env_state_data_q.put(self.g_state_data_queue)
        else:
            self.env_state_data_q.put(self.g_state_data_queue)



    def start_keyboard_listener(self):
        listener = keyboard.Listener(
            on_press=self.on_press)
        listener.start()

    def on_press(self, key):
        global fb_x, fb_y, fb_z, fb_pitch, fb_yaw, fb_roll, move_extractor_fl, cart_pos, cart_angle

        try:
            if key.char == '5':
                Thread(target=self.pusher.move_to_block_push, args=(5,)).start()
            if key.char == '6':
                Thread(target=self.pusher.move_to_block_push, args=(6,)).start()
            if key.char == '7':
                Thread(target=self.pusher.move_to_block_push, args=(7,)).start()
            if key.char == '8':
                Thread(target=self.pusher.move_to_block_push, args=(8,)).start()
            if key.char == '9':
                Thread(target=self.pusher.move_to_block_push, args=(9,)).start()
            if key.char == '+':
                self.extractor.move_in_direction('up')
            if key.char == '-':
                self.extractor.move_in_direction('down')
            if key.char == '.':
                self.pusher.move_pusher_to_next_block()
            if key.char == ',':
                self.pusher.move_pusher_to_previous_block()
            if key.char == 'p':
                # print(self.tower.get_poses_cv([3, 4, 5], False))
                self.toppled_fl = True
                # self.move_to_random_block()
            if key.char == 'q':
                self.set_screenshot_flag()
        except AttributeError:
            if key == keyboard.Key.up:
                self.pusher.move_pusher_in_direction('forward')
                # self.extractor.move_in_direction('forward')
            if key == keyboard.Key.down:
                self.pusher.move_pusher_in_direction('backwards')
                # self.extractor.move_in_direction('backwards')
            if key == keyboard.Key.left:
                self.pusher.move_pusher_in_direction('left')
                # self.extractor.move_in_direction('left')
            if key == keyboard.Key.right:
                self.pusher.move_pusher_in_direction('right')
                # self.extractor.move_in_direction('right')

    def take_screenshot(self):
        if self.on_screen_rendering:
            data = np.asarray(self.viewer.read_pixels(1920 - 66, 1080 - 55, depth=False)[::-1, :, :], dtype=np.uint8)
            data[:, :, [0, 2]] = data[:, :, [2, 0]]
            # return data
            cv2.imwrite('./screenshots/screenshot.png', data)
        else:
            # render camera #1

            self.viewer.render(1920 - 66, 1080 - 55, -1)
            data = np.asarray(self.viewer.read_pixels(1920 - 66, 1080 - 55, depth=False)[::-1, :, :], dtype=np.uint8)
            data[:, :, [0, 2]] = data[:, :, [2, 0]]

            # save data
            if data is not None:
                # cv2.imwrite("screenshots/offscreen1.png", data)
                return data

            # render camera #2
            self.viewer.render(1920 - 66, 1080 - 55, 1)
            data = np.asarray(self.viewer.read_pixels(1920 - 66, 1080 - 55, depth=False)[::-1, :, :], dtype=np.uint8)
            data[:, :, [0, 2]] = data[:, :, [2, 0]]

            # save data
            if data is not None:
                cv2.imwrite("screenshots/offscreen2.png", data)
                return data

    def set_screenshot_flag(self):
        self.screenshot_fl = True

    def sleep_simtime(self, t):
        current_time = self.t * g_timestep
        while self.t * g_timestep < current_time + t and self.simulation_running():
            time.sleep(0.05)

    def simulation_running(self):
        if self.tower_toppled() or self.exception_occurred() or self.timeout():
            return False
        else:
            return True

    def tower_toppled(self):
        if self.toppled_fl:
            return True
        else:
            return False

    def set_tower_toppled(self):
        self.toppled_fl = True

    def exception_occurred(self):
        if self.exception_fl:
            return True
        else:
            return False

    def timeout(self):
        if self.timeout_fl:
            return True
        else:
            return False

    def pause(self):
        self.pause_fl = True

    def resume(self):
        self.pause_fl = False

    def move_to_block(self, lvl, pos):
        log.debug(f"Before get positions")
        positions = self.tower.get_positions()
        orientations = self.tower.get_orientations()
        log.debug(f"After get positions")
        block_id = self.tower.get_block_id_from_pos(lvl, pos, positions, orientations, coordinate_axes_pos)
        log.debug(f"After get block id from pos")
        if block_id is not None:
            log.debug(f"Before pusher move to block")
            self.pusher.move_to_block(block_id)
            log.debug(f"After pusher move to block")
            self.current_block_id = block_id
        else:
            log.warning(f"There is no block on position ({lvl}, {pos}).")

    def push(self):
        log.debug(f"Push #1")
        force, block_displacement = self.pusher.push()
        self.total_distance += block_displacement
        log.debug(f"Push #2")

        return force, block_displacement

    def pull_and_place(self, lvl, pos):
        log.debug(f"Pull and place #1")
        # set the flag, so that the tilt is not computing during placing
        self.placing_fl = True

        id = 3 * lvl + pos
        self.extractor.extract_and_put_on_top_using_magic(id)
        log.debug(f"Pull and place #2")

    def move_to_random_block(self):
        log.debug(f"Move to random block #1")
        # get only legal block positions (check only the blocks that are below the highest full layer)
        positions = self.tower.get_positions()
        log.debug(f"After get positions")
        layers = self.tower.get_layers(positions)
        log.debug(f"After get layers")
        if len(layers[-1]) == 3:
            offset_from_top = 1
        else:
            offset_from_top = 2
        max_lvl = len(layers) - offset_from_top
        available_lvls = {lvl: self.checked_positions[lvl] for lvl in self.checked_positions if lvl < max_lvl}

        log.debug(f"Move to random block #2")
        if not available_lvls:
            return False

        # randomly choose one position
        lvl = random.choice(list(available_lvls.keys()))
        pos = random.sample(self.checked_positions[lvl], 1)[0]

        # update global variables
        self.current_lvl = lvl
        self.current_lvl_pos = pos

        # remove position and lvl if it has no unchecked positions
        self.checked_positions[lvl].remove(pos)
        if not self.checked_positions[lvl]:
            del self.checked_positions[lvl]

        log.debug(f"Before move to block")
        self.move_to_block(lvl, pos)
        log.debug(f"After move to block")

        log.debug(f"Move to random block #3")
        return True

    def get_state_advanced(self, new_block, force=0, block_displacement=0, mode=0):
        """
        mode=0: return gym compatible state
        mode=1: return state as State class
        """

        if mode == 0:

            log.debug(f"Get state#1")

            # pause simulation
            self.pause()

            # calculate side
            side = None
            if self.current_block_id % 6 < 3:
                side = 0
            else:
                side = 1

            # get block positions
            log.debug(f"Before get positions")
            block_positions = self.tower.get_positions()
            block_orientations = self.tower.get_orientations()
            log.debug(f"After get positions")

            # calculate tilt
            log.debug(f"Before get tilt")
            tilt_2ax = self.tower.get_tilt_2ax(block_positions, block_orientations, self.current_block_id)
            log.debug(f"After get tilt")
            if new_block:
                self.initial_tilt_2ax = tilt_2ax
                self.previous_step_displacement = np.array([0, 0])
                self.previous_step_z_rot = 0
                self.positions_from_last_step = copy.deepcopy(block_positions)
            last_tilt_2ax = tilt_2ax - self.initial_tilt_2ax

            # get last z_rotation
            log.debug(f"Before get z rotaiton")
            z_rotation_last = self.tower.get_last_z_rotation(self.pusher.current_block, block_orientations)
            log.debug(f"After get z rotaiton")

            # get current round displacement
            current_round_displacement = self.tower.get_last_displacement_2ax(self.pusher.current_block, block_positions, block_orientations)
            log.debug(f"After get displacement")

            # get last step tower displacement and z rotation
            current_step_tower_displacement = current_round_displacement - self.previous_step_displacement
            current_step_z_rot = z_rotation_last - self.previous_step_z_rot

            # reset previous displacement and z rotation
            self.previous_step_displacement = current_round_displacement
            self.previous_step_z_rot = z_rotation_last

            total_block_distance = self.total_distance

            # get block height
            block_height = block_positions[self.current_block_id][2] / one_millimeter

            # get layer configuration
            # 0) □□□
            # 1) □x□
            # 2) □□x
            # 3) x□□
            # 4) x□x
            log.debug(f"Before get layers state")
            layers = self.tower.get_layers_state(block_positions, block_orientations, coordinate_axes_pos)
            log.debug(f"After get layers state")
            current_layer = layers[self.current_lvl]
            print(f"Current layer: {current_layer}")
            layer_configuration = 0
            if current_layer[0] is not None and current_layer[1] is not None and current_layer[2] is not None:
                layer_configuration = 0
            elif current_layer[0] is not None and current_layer[2] is not None:
                layer_configuration = 1
            elif current_layer[0] is not None and current_layer[1] is not None:
                layer_configuration = 2
            elif current_layer[1] is not None and current_layer[2] is not None:
                layer_configuration = 3
            elif current_layer[1] is not None:
                layer_configuration = 4

            # resume simulation
            self.resume()

            # normalize state
            state = np.array([force, block_displacement, total_block_distance, current_step_tower_displacement[0],
                             current_step_tower_displacement[1], current_round_displacement[0],
                             current_round_displacement[1], last_tilt_2ax[0], last_tilt_2ax[1], current_step_z_rot,
                             z_rotation_last, side, block_height, self.current_lvl_pos, layer_configuration])
            log.debug(f"Get state#2")

            return state

        if mode == 1:
            block_positions = self.tower.get_positions()

            start_time_total = time.time()
            start_time = time.time()
            tilt = self.tower.get_tilt_2ax(block_positions, self.current_block_id)
            elapsed_time = time.time() - start_time
            # log.debug(f"Tilt time: {elapsed_time*1000:.2f}")

            start_time = time.time()
            displacement_total = self.tower.get_abs_displacement_2ax(self.pusher.current_block, block_positions)
            elapsed_time = time.time() - start_time
            # log.debug(f"Abs displacement time: {elapsed_time*1000:.2f}")

            start_time = time.time()
            current_round_displacement = self.tower.get_last_displacement_2ax(self.pusher.current_block, block_positions)
            elapsed_time = time.time() - start_time
            # log.debug(f"Last displacement time: {elapsed_time*1000:.2f}")

            start_time = time.time()
            z_rotation_total = self.tower.get_total_z_rotation(self.pusher.current_block)
            elapsed_time = time.time() - start_time
            # log.debug(f"Total rotation time: {elapsed_time*1000:.2f}")

            start_time = time.time()
            z_rotation_last = self.tower.get_last_z_rotation(self.pusher.current_block)
            elapsed_time = time.time() - start_time
            # log.debug(f"Last rotation time: {elapsed_time*1000:.2f}")

            status = Status.pushing
            self.total_distance += block_displacement

            start = time.time()
            tower_state = self.tower.get_layers_state(block_positions)
            elapsed = time.time() - start
            # log.debug(f"Layers state time: {elapsed*1000:.2f}")

            elapsed_time_total = time.time() - start_time_total
            # log.debug(f"Elapsed time total: {elapsed_time_total*1000:.2f}")

            state = State(tower_state, force, block_displacement, self.total_distance, tilt,
                          displacement_total, current_round_displacement, z_rotation_total, z_rotation_last, status)

            log.debug(f"{state}")

    def get_state(self, new_block, force=0, block_displacement=0, mode=0):
        """
        mode=0: return gym compatible state
        mode=1: return state as State class
        """

        if mode == 0:

            log.debug(f"Get state#1")

            # pause simulation
            self.pause()

            # calculate side
            side = None
            if self.current_block_id % 6 < 3:
                side = 0
            else:
                side = 1

            # get block positions
            log.debug(f"Before get positions")
            block_positions = self.tower.get_positions()
            block_orientations = self.tower.get_orientations()
            log.debug(f"After get positions")

            # calculate tilt
            log.debug(f"Before get tilt")
            tilt_2ax = self.tower.get_tilt_2ax(block_positions, block_orientations, self.current_block_id)
            log.debug(f"After get tilt")
            if new_block:
                self.initial_tilt_2ax = tilt_2ax
                self.previous_step_displacement = np.array([0, 0])
                self.previous_step_z_rot = 0
                self.positions_from_last_step = copy.deepcopy(block_positions)
            last_tilt_2ax = tilt_2ax - self.initial_tilt_2ax

            # get last z_rotation
            log.debug(f"Before get z rotaiton")
            z_rotation_last = self.tower.get_last_z_rotation(self.pusher.current_block, block_orientations)
            log.debug(f"After get z rotaiton")

            # get current round displacement
            current_round_displacement = self.tower.get_last_displacement_2ax(self.pusher.current_block, block_positions, block_orientations)
            log.debug(f"After get displacement")

            # get max displacement
            current_round_block_max_displacement = self.tower.get_last_max_displacement_2ax(self.current_block_id,
                                                                                            block_positions,
                                                                                            block_orientations)

            # get last step tower displacement and z rotation
            current_step_tower_displacement = np.linalg.norm(
                self.get_last_step_avg_disp(block_positions, block_orientations))
            current_step_max_displacement = max(0, np.linalg.norm(
                self.get_last_step_max_disp(block_positions, block_orientations)) - 1)

            # get last step tower displacement and z rotation
            # current_step_tower_displacement = current_round_displacement - self.previous_step_displacement
            # current_step_z_rot = z_rotation_last - self.previous_step_z_rot

            # reset previous displacement and z rotation
            self.previous_step_displacement = current_round_displacement
            self.previous_step_z_rot = z_rotation_last

            total_block_distance = self.total_distance

            # get block height
            block_height = block_positions[self.current_block_id][2] / one_millimeter

            # get layer configuration
            # 0) □□□
            # 1) □x□
            # 2) □□x
            # 3) x□□
            # 4) x□x
            log.debug(f"Before get layers state")
            layers = self.tower.get_layers_state(block_positions, block_orientations, coordinate_axes_pos)
            log.debug(f"After get layers state")
            current_layer = layers[self.current_lvl]
            print(f"Current layer: {current_layer}")
            layer_configuration = 0
            if current_layer[0] is not None and current_layer[1] is not None and current_layer[2] is not None:
                layer_configuration = 0
            elif current_layer[0] is not None and current_layer[2] is not None:
                layer_configuration = 1
            elif current_layer[0] is not None and current_layer[1] is not None:
                layer_configuration = 2
            elif current_layer[1] is not None and current_layer[2] is not None:
                layer_configuration = 3
            elif current_layer[1] is not None:
                layer_configuration = 4

            # resume simulation
            self.resume()

            # normalize state
            state = np.array([force,
                              block_displacement,
                              total_block_distance,
                              current_step_tower_displacement,
                              np.linalg.norm(current_round_displacement),
                              current_step_max_displacement,
                              block_height,
                              self.current_lvl_pos,
                              layer_configuration])
            log.debug(f"Get state#2")

            return state

        if mode == 1:
            block_positions = self.tower.get_positions()

            start_time_total = time.time()
            start_time = time.time()
            tilt = self.tower.get_tilt_2ax(block_positions, self.current_block_id)
            elapsed_time = time.time() - start_time
            # log.debug(f"Tilt time: {elapsed_time*1000:.2f}")

            start_time = time.time()
            displacement_total = self.tower.get_abs_displacement_2ax(self.pusher.current_block, block_positions)
            elapsed_time = time.time() - start_time
            # log.debug(f"Abs displacement time: {elapsed_time*1000:.2f}")

            start_time = time.time()
            current_round_displacement = self.tower.get_last_displacement_2ax(self.pusher.current_block, block_positions)
            elapsed_time = time.time() - start_time
            # log.debug(f"Last displacement time: {elapsed_time*1000:.2f}")

            start_time = time.time()
            z_rotation_total = self.tower.get_total_z_rotation(self.pusher.current_block)
            elapsed_time = time.time() - start_time
            # log.debug(f"Total rotation time: {elapsed_time*1000:.2f}")

            start_time = time.time()
            z_rotation_last = self.tower.get_last_z_rotation(self.pusher.current_block)
            elapsed_time = time.time() - start_time
            # log.debug(f"Last rotation time: {elapsed_time*1000:.2f}")

            status = Status.pushing
            self.total_distance += block_displacement

            start = time.time()
            tower_state = self.tower.get_layers_state(block_positions)
            elapsed = time.time() - start
            # log.debug(f"Layers state time: {elapsed*1000:.2f}")

            elapsed_time_total = time.time() - start_time_total
            # log.debug(f"Elapsed time total: {elapsed_time_total*1000:.2f}")

            state = State(tower_state, force, block_displacement, self.total_distance, tilt,
                          displacement_total, current_round_displacement, z_rotation_total, z_rotation_last, status)

            log.debug(f"{state}")

    def get_last_step_avg_disp(self, positions, orientations):
        return self.tower._get_displacement_2ax(self.current_block_id, self.positions_from_last_step, positions, orientations)

    def get_last_step_max_disp(self, positions, orientations):
        return self.tower._get_last_max_displacement_2ax(self.current_block_id, self.positions_from_last_step, positions, orientations)

    def compute_reward_old(self, block_displacement, tower_displacement, tower_z_rotation, tower_tilt):
        tower_displacement_1ax = np.linalg.norm(tower_displacement)
        tower_tilt_1ax = np.linalg.norm(tower_tilt)
        z_rot = abs(tower_z_rotation)
        coefficients = np.array([1, -1, -1, -1])
        reward = sum(coefficients * np.array([block_displacement, tower_displacement_1ax, tower_tilt_1ax, z_rot]))

        return reward

    def compute_reward_old2(self, state, normalize):
        block_displacement = state[1]
        tilt = state[7:9]
        current_step_tower_displacement = state[3:5]
        current_step_tower_z_rotation = state[9]

        tower_displacement_1ax = np.linalg.norm(current_step_tower_displacement)
        tower_tilt_1ax = np.linalg.norm(tilt)
        z_rot = abs(current_step_tower_z_rotation)
        coefficients = np.array([1, -1.5, -1, -2])

        reward = sum(coefficients * np.array([block_displacement, tower_displacement_1ax, tower_tilt_1ax, z_rot]))

        # normalize reward
        if normalize:
            reward = self.normalize_reward(reward)

        return reward

    def compute_reward(self, state, normalize):
        block_displacement = state[1]
        current_step_tower_displacement = state[3]
        current_step_max_displacement = state[5]

        tower_displacement_1ax = np.linalg.norm(current_step_tower_displacement)
        coefficients = np.array([1, -6, -3])

        reward = sum(coefficients * np.array([block_displacement, tower_displacement_1ax, current_step_max_displacement]))

        # normalize reward
        if normalize:
            reward = self.normalize_reward(reward)

        return reward

    def normalize_state(self, state):
        return (state - state_space_means) / state_space_stds

    def normalize_reward(self, reward):
        return (reward - reward_mean) / reward_std

    # returns observation, reward, done, info
    def step(self, action, normalize):
        log.debug(f"Jenga step#1")
        if self.exception_occurred() and self.tower_toppled():
            return self.last_state, self.last_reward, True, {'exception': True, 'toppled': True, 'timeout': False, 'last_screenshot': self.last_screenshot, 'extracted_blocks': self.extracted_blocks}

        if self.exception_occurred():
            return self.last_state, self.last_reward, True, {'exception': True, 'toppled': False, 'timeout': False, 'last_screenshot': self.last_screenshot, 'extracted_blocks': self.extracted_blocks}

        if self.tower_toppled():
            return self.last_state, tower_toppled_reward, True, {'exception': False, 'toppled': True, 'timeout': False, 'last_screenshot': self.last_screenshot, 'extracted_blocks': self.extracted_blocks}
        log.debug(f"Jenga step#2")
        if action == 0:
            log.debug(f"Jenga step#3")
            # reset state variables
            self.total_distance = 0
            self.steps_pushed = 0
            log.debug(f"Before move to random block")
            res = self.move_to_random_block()
            log.debug(f"After move to random block")
            done = not res
            reward = 0
            info = {'exception': False, 'toppled': False, 'timeout': False, 'last_screenshot': self.last_screenshot, 'extracted_blocks': self.extracted_blocks}
            log.debug(f"Before get state!")
            state = self.get_state(new_block=True)
            log.debug(f"After get state!")

            # normalize state
            if normalize:
                state = self.normalize_state(state)

            # save the last state and reward
            self.last_state = state
            self.last_reward = reward

            # wait until tower stabilizes
            self.sleep_simtime(2)
            log.debug(f"Jenga step#4")

            if self.plot_env_state:
                self.update_env_state_plot(state, reward)

            return state, reward, done, info

        if action == 1:
            log.debug(f"Jenga step#5")
            # extract and move to the next block
            if self.steps_pushed == self.max_pushing_distance:
                log.debug(f"Jenga step#6")
                # reset state variables
                self.total_distance = 0
                self.steps_pushed = 0

                log.debug(f"Before pull_and_place")
                self.pull_and_place(self.current_block_id//3, self.current_block_id % 3)
                log.debug(f"After pull_and_place")


                log.debug(f"Before move_to_random_block")
                res = self.move_to_random_block()
                log.debug(f"After move_to_random_block")
                self.extracted_blocks += 1
                done = not res
                reward = reward_extract
                info = {'exception': False, 'toppled': False, 'timeout': False, 'last_screenshot': self.last_screenshot, 'extracted_blocks': self.extracted_blocks}

                log.debug(f"Before  get state")
                state = self.get_state(new_block=True)
                log.debug(f"After  get state")

                # normalize state
                if normalize:
                    state = self.normalize_state(state)

                # save the last state and reward
                self.last_state = state
                self.last_reward = reward

                # wait until tower stabilizes
                self.sleep_simtime(2)

                if self.plot_env_state:
                    self.update_env_state_plot(state, reward)

                log.debug(f"Jenga step#7")
                return state, reward, done, info

            else:  # push
                # start_index = 7
                # block_quat = Quaternion(self.tower.get_orientation(self.current_block_id))
                # block_pos = self.tower.get_position(self.current_block_id)
                #
                # # calculate pusher orientation
                # first_block_end = block_pos + block_quat.rotate(x_unit_vector) * block_length_mean / 2
                # second_block_end = block_pos + block_quat.rotate(-x_unit_vector) * block_length_mean / 2
                # first_distance = np.linalg.norm(coordinate_axes_pos - first_block_end)
                # second_distance = np.linalg.norm(coordinate_axes_pos - second_block_end)
                # if first_distance > second_distance:
                #     offset_quat = block_quat
                # else:
                #     offset_quat = block_quat * Quaternion(axis=[0, 0, 1], degrees=180)
                #
                # new_pos = block_pos - offset_quat.rotate(x_unit_vector) * 32 * one_millimeter
                # l = list(new_pos) + list(block_quat)
                # self.sim.data.qpos[start_index + self.current_block_id * 7:start_index + (self.current_block_id + 1) * 7] = l
                # state = self.get_state(new_block=False, force=0, block_displacement=0)
                # info = {'exception': False, 'toppled': False}
                # self.steps_pushed += 1
                # return state, 0, False, info

                log.debug(f"Jenga step#8")
                force, displacement = self.push()
                log.debug(f"Pushed!")
                self.steps_pushed += 1
                log.debug(f"Before get state")
                state = self.get_state(new_block=False, force=force, block_displacement=displacement)
                log.debug(f"After get state")


                # np.array([force, block_distance, total_block_distance, current_step_tower_displacement[0],
                #           current_step_tower_displacement[1], current_round_displacement[0],
                #           current_round_displacement[1], last_tilt_2ax[0], last_tilt_2ax[1], current_step_z_rot,
                #           z_rotation_last, side, block_height])

                reward = self.compute_reward(state, normalize)
                done = False
                info = {'exception': False, 'toppled': False, 'timeout': False, 'last_screenshot': self.last_screenshot, 'extracted_blocks': self.extracted_blocks}

                # normalize state
                if normalize:
                    state = self.normalize_state(state)

                # save the last state and reward
                self.last_state = state
                self.last_reward = reward

                log.debug(f"Jenga step#9")

                if self.plot_env_state:
                    self.update_env_state_plot(state, reward)

                return state, reward, done, info

    def debug_move_to_zero(self):
        self.extractor.set_position([10, 0, 1])

    def debug_collect_localization_errors(self):
        true_positions = self.tower.get_positions()
        true_orientations = self.tower.get_orientations()
        estimated_poses, _, _ = self.tower.get_poses_cv_mujoco(return_images=True)
        log.debug(estimated_poses)
        estimated_positions = {x: estimated_poses[x]['pos'] for x in estimated_poses}
        estimated_orientations = {x: estimated_poses[x]['orientation'] for x in estimated_poses}
        tags_detected = {x: estimated_poses[x]['tags_detected'] for x in estimated_poses}

        for i in range(g_blocks_num):
            # if at least one tag was detected
            if i in estimated_positions:
                position_error = (true_positions[i] - estimated_positions[i]) / one_millimeter
                orientation_error = get_angle_between_quaternions_3ax(true_orientations[i], estimated_orientations[i])
                quaternion_error = true_orientations[i] - estimated_orientations[i]
                self.localization_errors.append(list(position_error) + list(orientation_error) + list(tags_detected[i]) + [i] + [true_positions[i][2]] + list(quaternion_error))

    def debug_write_collected_errors(self):
        log.debug(f"{self.localization_errors}")
        with open(f'localization_errors_{os.getpid()}.csv', mode='w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.localization_errors)

    def abort_simulation(self):
        self.simulation_aborted = True
        while not self.simulation_over:
            time.sleep(0.1)

    def get_seed(self):
        return self.seed

    def get_sim_time(self):
        return self.t * g_timestep

    def get_last_screenshot(self):
        return self.last_screenshot

    def simulate(self):
        log.debug(f"Simulation thread running!")
        # initialize queue for timings
        timings = collections.deque(maxlen=100)


        # initialize viewer
        if self.render:
            if self.on_screen_rendering:
                self.viewer = MjViewer(self.sim)
            else:
                self.viewer = MjRenderContextOffscreen(self.sim)
        else:
            self.viewer = MjRenderContextOffscreen(self.sim)

        # start with specific camera position
        self.viewer.cam.elevation = -37
        self.viewer.cam.azimuth = 130
        self.viewer.cam.lookat[0:3] = [-4.12962146, -4.90275717, 7.20812166]
        self.viewer._run_speed = 16
        self.viewer.cam.distance = 25

        # initializing step
        # this is needed to initialize positions and orientations of the objects in the scene
        self.sim.step()

        # initialize internal objects
        if self.render:
            self.tower = Tower(self.sim, self.viewer)
        else:
            self.tower = Tower(self.sim, None)
        self.pusher = Pusher(self.sim, self.tower, self)
        self.extractor = Extractor(self.sim, self.tower, self)

        # create histogram plot for position errors
        x_errors = np.arange(g_blocks_num)
        y_errors = np.zeros(g_blocks_num)
        plt.ion()
        fig_errors = plt.figure()
        ax = fig_errors.add_subplot(111)
        ax.set(ylim=(0, 15))
        ax.xaxis.set_ticks(range(g_blocks_num))
        ax.yaxis.set_ticks(range(10))
        line1 = ax.bar(x_errors, y_errors)
        ax.grid(zorder=0)

        log.debug(f"Initialization done")

        # debug_tower_toppled_time = int(random.random()*10000 + 1)
        # debug_tower_toppled_time = 10000

        # simulation loop
        try:
            while not self.tower_toppled() and not self.simulation_aborted:
                if not self.pause_fl:
                    self.t += 1


                    # update positions of the pusher and gripper
                    self.pusher.update_position(self.t)
                    self.extractor.update_positions(self.t)

                    # make step
                    cycle_start_time = time.time()
                    self.sim.step()
                    cycle_elapsed_time = time.time() - cycle_start_time
                    timings.append(cycle_elapsed_time)

                    # check if tower is toppled
                    if self.t % 100 == 0 and self.tower.toppled(self.tower.get_positions(), self.current_block_id):
                        self.toppled_fl = True
                        log.debug(f"Tower toppled!")

                    # if self.t == debug_tower_toppled_time:
                    #     self.toppled_fl = True
                    #     log.debug(f"Fake tower toppled!")

                    # get positions of blocks and plot images
                    if self.t % 100 == 0 and self.render and not self.on_screen_rendering:
                        # self.off_screen_render_and_plot_errors(line1, fig_errors)
                        pass

                    if self.t % 100 == 0 and self.render:
                        self.debug_collect_localization_errors()
                        pass

                    # render if on screen rendering
                    if self.render and self.on_screen_rendering:
                        self.viewer.render()

                    # take a screenshot if q was pressed
                    if self.screenshot_fl:
                        self.screenshot_fl = False
                        self.take_screenshot()

                    # take regular screenshots for inspection
                    if self.t % 200 == 0:
                        self.last_screenshot = self.take_screenshot()

                    # update force sensor data if enabled
                    if self.plot_force:
                        self.update_force_sensor_plot()


                    # log.debug(f"Cycle time: {np.mean(timings)*1000}ms")
                    if self.t > 100 and os.getenv('TESTING') is not None:
                        break
                else:
                    time.sleep(0.001)
            log.debug(f"Exit try!")
        except Exception:
            log.exception(f"Exception occured!")
            traceback.print_exc()
            self.exception_fl = True

        self.simulation_over = True
        # self.debug_write_collected_errors()
        log.debug(f"Exit simulation thread!")


def check_all_blocks(simulation):
    start_time = time.time()
    simulation.move_to_block_push(0, 0)
    time.sleep(1)
    loose_blocks = []
    force_threshold = 240000 * 0.4
    angle_threshold = 3
    exception_occurred = False
    tower_toppled = False
    timeout = False
    for i in range(g_blocks_num - 9):
        total_displacement = 0
        fl = 0

        if i % 3 == 0:
            force_threshold -= 10500 * 0.4

        for j in range(45):
            # simulation.debug_move_to_zero()
            exception_occurred = simulation.exception_occurred()
            tower_toppled = simulation.tower_toppled()
            timeout = simulation.timeout()
            if exception_occurred or tower_toppled or timeout:
                fl = 1
                break
            state = simulation.push()
            force = state.force
            displacement = state.block_distance
            tilt = np.linalg.norm(state.tilt)
            total_displacement += displacement
            if force > force_threshold or tilt > angle_threshold:
                fl = 1
                break

        if tower_toppled:
            log.debug(f"Tower toppled!")
            break
        if exception_occurred:
            log.debug(f"Simulation has become unstable")
            break
        if timeout:
            log.debug(f"Timeout!")
            break
        if fl == 0:
            loose_blocks.append(i)
            simulation.pull_and_place(i // 3, i % 3)

        log.debug(f"Extracted blocks: {loose_blocks}")

        simulation.move_to_block_push((i + 1) // 3, (i + 1) % 3)
        time.sleep(1)

    error = None
    if tower_toppled:
        error = Error.tower_toppled
    if exception_occurred:
        error = Error.exception_occurred
    if timeout:
        error = Error.timeout

    screenshot = simulation.get_last_screenshot()

    real_elapsed_time = time.time() - start_time

    simulation.debug_write_collected_errors()

    return SimulationResult(extracted_blocks=loose_blocks,
                            real_time=real_elapsed_time,
                            sim_time=simulation.get_sim_time(),
                            error=error,
                            seed=simulation.get_seed(),
                            screenshot=screenshot)


def run_one_simulation(render=True, timeout=600, seed=None):
    env = jenga_env(render=render, timeout=timeout, seed=seed)
    try:
        # res = check_all_blocks(env)
        for i in range(1000):
            time.sleep(1)
        env.reset()

        for i in range(10):
            time.sleep(1)
    except Exception as e:
        log.error(f"Exception in the algorithm occurred!")
        traceback.print_exc()
        res = None
    return res


class jenga_env_wrapper(gym.Env):
    action_space = gym.spaces.Discrete(2)
    high = np.array([8.602395379871112, 1.5511497626226662, 2.127394223313028, 9.25020015965446, 13.45107554958695,
                     3.7951169685753365, 3.054330633651127, 3.649017183775844, 1.5757055313785495, 17.312344327442563,
                     4.71880866755271, 1.451065147438874, 1.8358448244069085, 2, 4])
    low = np.array(
        [-0.41973308139854426, -1.6503939248719546, -1.070771476066865, -7.326824408306893, -10.447351338602731,
         -2.9554260815691626, -3.5417146920008684, -4.731617156055343, -3.372264772803644, -6.205548486206324,
         -1.5037625212933003, -0.6891489343293768, -1.1772412151405776, 0, 0])
    observation_space = gym.spaces.Box(low=high, high=high)

    def __init__(self, normalize, seed=None):
        log.debug(f"Start initialization")
        self.action_q = Queue()
        self.state_q = Queue()
        self.process_running_q = Queue()
        self.get_state_q = Queue()
        self.process = None
        self.action_counter = 0
        self.normalize = normalize
        self.seed = seed
        self.sim_pid = None

        # variable for the last state
        self.last_response = None
        log.debug(f"Finish initialization")

    def env_process(self, action_q: Queue, state_q: Queue, process_running_q: Queue, get_state_q: Queue, normalize, seed):
        log.debug(f"Start process")
        env = jenga_env(render=True, seed=seed)
        log.debug(f"Env created")
        flag = True
        debug_collapse_time = random.random()*30 + 3
        start_time = time.time()
        while flag and process_running_q.empty():
            # if time.time() - start_time > debug_collapse_time:
            #     env.debug_move_to_zero()

            if not action_q.empty():
                log.debug("Loop #1")
                action = action_q.get()
                log.debug(f"Action: {action}")
                obs, reward, done, info = env.step(action, normalize=normalize)
                log.debug(f"After step!")
                state_q.put((obs, reward, done, info))
                if info['exception'] or info['toppled']:
                    if info['exception']:
                        log.exception(f"Exception occurred!")
                    if info['toppled']:
                        log.info(f"Tower toppled!")
                    flag = False
            if not get_state_q.empty():
                log.debug(f"Loop #2")
                state = env.get_state(new_block=True)
                state_q.put(state)
                log.debug(f"Before get state")
                get_state_q.get()
                log.debug(f"After get state")
            time.sleep(0.1)

            if env.exception_occurred() or env.tower_toppled():
                if env.exception_occurred():
                    log.exception(f"Exception occurred!")
                    info['exception'] = True
                if env.tower_toppled():
                    info['toppled'] = True
                    log.info(f"Tower toppled!")
                state_q.put((obs, reward, done, info))
                flag = False

        if not process_running_q.empty():
            log.debug(f"Abort simulation from wrapper!")
            env.abort_simulation()
        log.debug(f"Flag: {flag}, process_running_q.empty() == {process_running_q.empty()}")
        log.debug(f"action_q.empty() == {action_q.empty()}")
        log.debug(f"Process ends!")

    def step(self, action):
        log.debug(f'Step #1')
        log.debug(f"Action # {self.action_counter}: {action}")
        self.action_counter += 1
        self.action_q.put(action)
        try:
            log.debug(f"Before get state")
            res = self.state_q.get(timeout=timeout_step)
            log.debug(f"After get state")
            self.last_response = res
            # print(res)
            log.debug(f"Step #2")
            return res
        except:
            # stop simulation
            self.process.terminate()
            log.exception("Timeout occurred!")
            obs = self.last_response[0]
            reward = self.last_response[1]
            info = self.last_response[3]
            info['timeout'] = True  # update info
            done = True  # update done
            log.debug(f"Step #3")
            return obs, reward, done, info

    def reset(self):
        log.debug(f"Reset #1")
        if self.process is not None:
            log.debug(f"Before terminate")
            self.process.terminate()
            log.debug(f"After terminate")
            while self.process.is_alive():
                log.debug('Wait for process exit!')
                time.sleep(0.1)
        log.debug(f"After processor running wait!")

        # reset object variables
        self.__init__(self.normalize, self.seed)

        log.debug(f"Before start new process")
        self.start_new_process()
        log.debug(f"SIM_PID:{self.sim_pid}:After start new process")
        self.get_state_q.put(1)
        log.debug(f"SIM_PID:{self.sim_pid}:Before get state")
        state = self.state_q.get()
        log.debug(f"SIM_PID:{self.sim_pid}:After get state")
        log.debug(f"SIM_PID:{self.sim_pid}:Reset #2")
        return state

    def close(self):
        # log.debug(f"SIM_PID:{self.sim_pid}:Close!")
        # if self.process.is_alive():
        #     self.process_running_q.put(1)
        #     log.debug(f"SIM_PID:{self.sim_pid}:Before wait")
        #     while self.process.is_alive():
        #         time.sleep(0.1)
        #     log.debug(f"SIM_PID:{self.sim_pid}:After wait")
        #     self.process_running_q.get(timeout=timeout_step)
        #     log.debug(f"SIM_PID:{self.sim_pid}:After process stopping")
        # log.debug(f"SIM_PID:{self.sim_pid}:Closed!")

        log.debug(f"SIM_PID:{self.sim_pid}:Close!")
        if self.process.is_alive():
            self.process.terminate()
            log.debug(f"SIM_PID:{self.sim_pid}:Before wait")
            while self.process.is_alive():
                time.sleep(0.1)
            log.debug(f"SIM_PID:{self.sim_pid}:After wait")
            log.debug(f"SIM_PID:{self.sim_pid}:After process stopping")
        log.debug(f"SIM_PID:{self.sim_pid}:Closed!")

    def process_func_debug(self):

        print(f"Process started!!!")
        log.debug(f"Process started!!!")
        # print(f"Process started2!!!")
        # foo()
        time.sleep(10)

    def start_new_process(self):
        log.debug(f"Start new process #1")
        self.process = Process(target=self.env_process, args=(self.action_q, self.state_q, self.process_running_q, self.get_state_q, self.normalize, self.seed))
        log.debug(f"Start new process #2")
        self.process.start()
        self.sim_pid = self.process.pid
        log.debug(f"SIM_PID:{self.sim_pid}:Start new process #3")

    def get_pid(self):
        return self.sim_pid


if __name__ == "__main__":

    env = jenga_env()

    env.pusher.move_along_own_axis('x', 10)

    for i in range(100):
        time.sleep(1)

    pass

    # start_total_time = time.time()
    #
    #
    # # params
    # N = 1
    # TOTAL = 1
    # timeout = 10  # in seconds
    # render = True
    # seed = None
    #
    #
    # process = Process(target=run_one_simulation, args=(render, timeout, seed))
    # process.start()
    #
    # while True:
    #     time.sleep(1)
    #
    #
    # all_results = [None for i in range(TOTAL)]
    # if N == -1:
    #     pool = Pool(maxtasksperchild=1)
    # else:
    #     pool = Pool(N, maxtasksperchild=1)
    # f = open('final_results.log', mode='w')
    #
    # # start the execution using a process pool
    # for i in range(TOTAL):
    #     all_results[i] = pool.apply_async(func=run_one_simulation, args=(render, timeout, seed))
    #
    # # print results while running
    # while any(list(map(lambda x: not x.ready(), all_results))):
    #     log.debug(f"######## Intermediate results #########")
    #     for i in range(TOTAL):
    #         if all_results[i].ready():
    #             res = all_results[i].get()
    #             log.debug(str(res))
    #     log.debug(f"#######################################")
    #     time.sleep(60)
    #
    #
    # # print final results:
    # log.debug(f"######## Final results #########")
    # for i in range(TOTAL):
    #     r = all_results[i].get()
    #     if r is not None:
    #         log.debug(f"#{i} " + str(r))
    #         f.write(f"#{i} " + str(r) + "\n")
    #         cv2.imwrite(f'./screenshots/res_scr#{i}.png', r.screenshot)
    #     else:
    #         log.debug(f"#{i} None")
    #         f.write(f"#{i} None\n")
    #
    # log.debug(f"################################")
    #
    # # calculate and print the total elapsed time
    # elapsed_total_time = time.time() - start_total_time
    # log.debug(f"Elapsed time in total: {int(elapsed_total_time)}s")
    # log.debug(f"Time per simulation: {int(elapsed_total_time/TOTAL)}s")

    # env = jenga_env_wrapper()
    #
    # env.reset()
    #
    # while True:
    #     time.sleep()

    # from stable_baselines import PPO2
    # from stable_baselines.common.policies import MlpPolicy
    #
    # model = PPO2(MlpPolicy, env, verbose=1)
    # model.learn(total_timesteps=200)

    # env.reset()
    #
    # for i in range(10):
    #     time.sleep(1)
    #     state, reward, done, info = env.step(1)
    #     print(state)
    #
    # env.reset()
    #
    # for i in range(10):
    #     time.sleep(1)
    #     state, reward, done, info = env.step(1)
    #     print(state)
    #
    # state, reward, done, info = env.step(0)
    # print(state)
    #
    # env.reset()


