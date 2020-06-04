from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen
from pynput import keyboard
import os
from generate_scene import generate_scene
import numpy as np
from pyquaternion import Quaternion
import time
from threading import Thread, Lock

# animated plot
import pyformulas as pf
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from constants import *
from pusher import Pusher
from tower import Tower
from extractor import Extractor
from math import sin, cos, radians
import cv2
import math
import logging
import colorlog
from simple_pid import PID
import gym
import collections
from multiprocessing import Process, Queue, Pool
from enum import Enum, auto
import cProfile
from pyinstrument import Profiler
import os
import traceback
import glfw
import psutil
import sys
from utils.utils import get_angle_between_quaternions_3ax
import csv
import random

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

log = logging.Logger("file_logger")
file_formatter = logging.Formatter('%(levelname)s:PID:%(process)d:%(funcName)s:%(message)s')
file_handler = logging.FileHandler(filename='jenga.log', mode='w')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)

class SimulationResult:
    def __init__(self, extracted_blocks, real_time, sim_time, error, seed, screenshot):
        self.extracted_blocks = extracted_blocks
        self.blocks_number = len(extracted_blocks)
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
                 status
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
        Status: {str(self.status)}"""

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
        self.on_screen_rendering = True
        self.plot_force = False
        self.screenshot_fl = False
        self.automatize_pusher = 1
        self.tower_toppled_fl = False
        self.simulation_aborted = False
        self.current_block_id = 0

        # globals
        self.last_screenshot = None

        # sensor data
        self.g_sensor_data_queue = []
        self.g_sensors_data_queue_maxsize = 250

        # initialize simulation
        scene_xml, self.seed = generate_scene(g_blocks_num, g_timestep, seed=seed)
        # pid = os.getpid()
        # f = open(f'scene_p#{pid}.txt', mode='w')
        # f.write(scene_xml)
        self.model = load_model_from_xml(scene_xml)
        self.sim = MjSim(self.model)
        self.pause_fl = False
        self.viewer = None
        self.t = 0
        self.available_blocks = [i for i in range(g_blocks_num-9)]

        # declare internal entities
        self.tower = None
        self.pusher = None
        self.extractor = None

        # initialize state variables
        self.total_distance = 0

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
        # self.start_keyboard_listener()

        # start force data plotting
        if self.plot_force:
            plotting_thread = Thread(target=self.plot_force_data, daemon=True)
            plotting_thread.start()

        # start simulation
        log.debug(f"Start simulation thread")
        self.simulation_thread = Thread(target=self.simulate, daemon=True)
        self.simulation_thread.start()

        log.debug(f"Simulation thread started!")

        # wait until pusher and extractor are initialized
        while self.pusher is None or self.extractor is None:
            time.sleep(0.1)

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

    def plot_force_data(self):
        fig = plt.figure()
        canvas = np.zeros((48, 64))
        screen = pf.screen(canvas, 'Force values')

        while True:
            fig.clear()
            ys = -np.array(self.g_sensor_data_queue)  # copy data because of no thread synchronisation
            xs = range(-len(ys) + 1, 1)
            plt.plot(xs, ys, c='black')
            fig.canvas.draw()
            stop = time.time()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            screen.update(image)

    def update_force_sensor_plot(self):
        current_sensor_value = self.sim.data.sensordata[g_blocks_num*3 + g_blocks_num*4]
        if len(self.g_sensor_data_queue) >= self.g_sensors_data_queue_maxsize:
            self.g_sensor_data_queue.pop(0)
        self.g_sensor_data_queue.append(current_sensor_value)

    def start_keyboard_listener(self):
        listener = keyboard.Listener(
            on_press=self.on_press)
        listener.start()

    def on_press(self, key):
        global fb_x, fb_y, fb_z, fb_pitch, fb_yaw, fb_roll, move_extractor_fl, cart_pos, cart_angle

        try:
            if key.char == '5':
                Thread(target=self.pusher.move_to_block, args=(5,)).start()
            if key.char == '6':
                Thread(target=self.pusher.move_to_block, args=(6,)).start()
            if key.char == '7':
                Thread(target=self.pusher.move_to_block, args=(7,)).start()
            if key.char == '8':
                Thread(target=self.pusher.move_to_block, args=(8,)).start()
            if key.char == '9':
                Thread(target=self.pusher.move_to_block, args=(9,)).start()
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
                self.debug_move_to_zero()
            if key.char == 'q':
                self.set_screenshot_flag()
        except AttributeError:
            if key == keyboard.Key.up:
                # pusher.move_pusher_in_direction('forward')
                self.extractor.move_in_direction('forward')
            if key == keyboard.Key.down:
                # pusher.move_pusher_in_direction('backwards')
                self.extractor.move_in_direction('backwards')
            if key == keyboard.Key.left:
                # pusher.move_pusher_in_direction('left')
                self.extractor.move_in_direction('left')
            if key == keyboard.Key.right:
                # pusher.move_pusher_in_direction('right')
                self.extractor.move_in_direction('right')

    def take_screenshot(self):
        if self.on_screen_rendering:
            data = np.asarray(self.viewer.read_pixels(1920 - 66, 1080 - 55, depth=False)[::-1, :, :], dtype=np.uint8)
            data[:, :, [0, 2]] = data[:, :, [2, 0]]
            return data
            # cv2.imwrite('./screenshots/screenshot.png', data)
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
        while self.t * g_timestep < current_time + t:
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

    def move_to_block(self, level, pos):
        log.debug(f"#1 Move")
        log.warning("Block Ids and positions mixing!")
        self.current_block_id = level*3 + pos
        block_id = 3 * level + pos
        self.pusher.move_to_block(block_id)
        force = 0
        block_displacement = 0
        block_positions = self.tower.get_positions()
        tilt = self.tower.get_tilt_2ax(block_positions, self.current_block_id)
        displacement_total = self.tower.get_abs_displacement_2ax(self.pusher.current_block, block_positions)
        last_displacement = 0
        z_rotation_total = self.tower.get_total_z_rotation(self.pusher.current_block)
        z_rotation_last = 0
        status = Status.ready_to_push
        self.total_distance = 0
        start_time = time.time()
        tower_state = self.tower.get_layers_state(block_positions)
        elapsed_time = time.time() - start_time
        state = State(tower_state, force, block_displacement, self.total_distance, tilt,
                      displacement_total, last_displacement, z_rotation_total, z_rotation_last, status)

        log.debug(f"#2 Move")

        return state

    def push(self):
        start_time = time.time()

        log.debug(f"#1 Push")

        force, block_displacement = self.pusher.push()

        # pause simulation
        self.pause()

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
        last_displacement = self.tower.get_last_displacement_2ax(self.pusher.current_block, block_positions)
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
                      displacement_total, last_displacement, z_rotation_total, z_rotation_last, status)

        log.debug(f"{state}")

        # resume simulation
        self.resume()
        # log.debug(f"#2 Push")
        return state

    def pull_and_place(self, lvl, pos):
        log.debug(f"#1 Pull")
        # set the flag, so that the tilt is not computing during placing
        self.placing_fl = True

        id = 3 * lvl + pos
        self.extractor.extract_and_put_on_top(id)
        force = 0
        block_displacement = 0
        block_positions = self.tower.get_positions()
        tilt = self.tower.get_tilt_2ax(block_positions, self.current_block_id)
        displacement_total = self.tower.get_abs_displacement_2ax(self.pusher.current_block, block_positions)
        last_displacement = 0
        z_rotation_total = self.tower.get_total_z_rotation(self.pusher.current_block)
        z_rotation_last = 0
        status = Status.ready
        self.total_distance = 0
        start_time = time.time()
        tower_state = self.tower.get_layers_state(block_positions)
        elapsed_time = time.time() - start_time
        state = State(tower_state, force, block_displacement, self.total_distance, tilt,
                      displacement_total, last_displacement, z_rotation_total, z_rotation_last, status)
        log.debug(f"#2 Pull")

        # reset the flag back
        self.placing_fl = False
        return state

    def move_to_random_block(self):
        if self.available_blocks:
            res = random.choice(self.available_blocks)
            self.available_blocks.remove(res)
            lvl = res // 3
            pos = res % 3
            self.move_to_block(lvl, pos)
            return True
        else:
            return False

    # returns observation, reward, done, info
    def step(self, action):
        if action == Action.move_to_block:
            raise NotImplementedError
        if action == Action.push:
            raise NotImplementedError
        if action == Action.pull_and_place:
            raise NotImplementedError

            

    def debug_move_to_zero(self):
        self.extractor.set_position([10, 0, 1])

    def debug_collect_localization_errors(self):
        true_positions = self.tower.get_positions()
        true_orientations = self.tower.get_orientations()
        estimated_poses, _, _ = self.tower.get_poses_cv(return_images=True)
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
        self.viewer._run_speed = 1
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
                    if self.t % 100 == 0 and not self.placing_fl and self.tower.get_tilt_1ax(self.tower.get_positions()) >= 15:
                        self.toppled_fl = True
                        log.debug(f"Tower toppled!")

                    # get positions of blocks and plot images
                    if self.t % 100 == 0 and self.render and not self.on_screen_rendering:
                        # self.off_screen_render_and_plot_errors(line1, fig_errors)
                        pass

                    if self.t % 100 == 0 and self.render:
                        # self.debug_collect_localization_errors()
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
            log.error(f"Exception occured!")
            traceback.print_exc()
            self.exception_fl = True

        # self.debug_write_collected_errors()
        log.debug(f"Exit simulation thread!")

def check_all_blocks(simulation):
    start_time = time.time()
    simulation.move_to_block(0, 0)
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

        simulation.move_to_block((i + 1) // 3, (i + 1) % 3)
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
        res = check_all_blocks(env)
    except Exception as e:
        log.error(f"Exception in the algorithm occurred!")
        traceback.print_exc()
        res = None
    return res



if __name__ == "__main__":

    start_total_time = time.time()

    # params
    N = 1
    TOTAL = 1
    timeout = 10  # in seconds
    render = True
    seed = None
    all_results = [None for i in range(TOTAL)]
    if N == -1:
        pool = Pool(maxtasksperchild=1)
    else:
        pool = Pool(N, maxtasksperchild=1)
    f = open('final_results.log', mode='w')

    # start the execution using a process pool
    for i in range(TOTAL):
        all_results[i] = pool.apply_async(func=run_one_simulation, args=(render, timeout, seed))

    # print results while running
    while any(list(map(lambda x: not x.ready(), all_results))):
        log.debug(f"######## Intermediate results #########")
        for i in range(TOTAL):
            if all_results[i].ready():
                res = all_results[i].get()
                log.debug(str(res))
        log.debug(f"#######################################")
        time.sleep(60)


    # print final results:
    log.debug(f"######## Final results #########")
    for i in range(TOTAL):
        r = all_results[i].get()
        if r is not None:
            log.debug(f"#{i} " + str(r))
            f.write(f"#{i} " + str(r) + "\n")
            cv2.imwrite(f'./screenshots/res_scr#{i}.png', r.screenshot)
        else:
            log.debug(f"#{i} None")
            f.write(f"#{i} None\n")

    log.debug(f"################################")

    # calculate and print the total elapsed time
    elapsed_total_time = time.time() - start_total_time
    log.debug(f"Elapsed time in total: {int(elapsed_total_time)}s")
    log.debug(f"Time per simulation: {int(elapsed_total_time/TOTAL)}s")


