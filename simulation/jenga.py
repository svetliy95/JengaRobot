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
from multiprocessing import Process, Queue
from enum import Enum, auto
import cProfile
from pyinstrument import Profiler
import os

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

class Error(Enum):
    tower_toppled = auto()
    exception_occurred = auto()
    timeout = auto()

class ActionType(Enum):
    push = auto()
    pull_and_place = auto()
    move_to_block = auto()

class Action:
    def __init__(self, type: ActionType, lvl, pos):
        assert type in [ActionType.push, ActionType.pull_and_place, ActionType.move_to_block], "Wrong action!"
        assert lvl in range(15), "Wrong level index!"
        assert pos in range(3), "Wrong position index!"
        self.type = type
        self.lvl = lvl
        self.pos = pos

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

    def __init__(self, render=True, timeout=600, seed=None):
        self.actions_q = Queue()
        self.state_q = Queue()
        self.exception_q = Queue()
        self.toppled_q = Queue()
        self.timeout_q = Queue()
        self.screenshot_q = Queue()
        self.abort_q = Queue()
        self.p = Process(target=self.process, args=(self.actions_q, render, timeout, seed), daemon=True)
        self.p.start()

    def process(self, q: Queue, render=True, timeout=600, seed=None):
        log.debug(f"Process #{os.getpid()} started with seed: {seed}")
        simulation_starting_time = time.time()
        # globals
        self.render = render
        self.on_screen_rendering = True
        self.plot_force = False
        self.screenshot_fl = False
        self.automatize_pusher = 1
        self.tower_toppled_fl = False
        self.simulation_aborted = False

        # sensor data
        self.g_sensor_data_queue = []
        self.g_sensors_data_queue_maxsize = 250

        # initialize simulation
        scene_xml = generate_scene(g_blocks_num, g_timestep, seed=seed)
        pid = os.getpid()
        f = open(f'scene_p#{pid}.txt', mode='w')
        f.write(scene_xml)
        self.model = load_model_from_xml(scene_xml)
        self.sim = MjSim(self.model)
        self.pause_fl = False
        self.t = 0

        # declare internal entities
        self.tower = None
        self.pusher = None
        self.extractor = None

        # initialize state variables
        self.total_distance = 0

        # create histogram plot for position errors
        self.x_errors = np.arange(g_blocks_num)
        self.y_errors = np.zeros(g_blocks_num)
        plt.ion()
        self.fig_errors = plt.figure()
        self.ax = self.fig_errors.add_subplot(111)
        self.ax.set(ylim=(0, 15))
        self.ax.xaxis.set_ticks(range(g_blocks_num))
        self.ax.yaxis.set_ticks(range(10))
        self.line1 = self.ax.bar(self.x_errors, self.y_errors)
        self.ax.grid(zorder=0)

        # start keyboard listener
        self.start_keyboard_listener()

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

        # listen for actions
        while self.simulation_thread.is_alive():
            if not q.empty():
                action = q.get()
                if action.type == ActionType.push:
                    state = self._push()
                    self.state_q.put(state)
                if action.type == ActionType.pull_and_place:
                    state = self._pull_and_place(action.lvl, action.pos)
                    self.state_q.put(state)
                if action.type == ActionType.move_to_block:
                    state = self._move_to_block(action.lvl, action.pos)
                    self.state_q.put(state)
            if not self.abort_q.empty():
                self.simulation_aborted = True
            if time.time() - simulation_starting_time > timeout:
                # abort the simulation
                self.timeout_q.put(Error.timeout)
                self._abort_simulation()

            time.sleep(0.01)

    def off_screen_render_and_plot_errors(self):
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
                self.line1[i].set_height(pos_error)
                if est_pose['tags_detected'] == 2:
                    self.line1[i].set_color('b')
                else:
                    self.line1[i].set_color('y')
            else:
                self.line1[i].set_height(10)
                self.line1[i].set_color('r')
        self.fig_errors.canvas.draw()
        self.fig_errors.canvas.flush_events()

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
                positions = self.tower.get_positions()

                start = time.time()
                for i in range(1000):
                    layers = self.tower.get_tilt_2ax(positions)
                log.debug(f"Layers: {layers}")
                elapsed = time.time() - start
                log.debug(f"100 tilt_2ax times = {elapsed /1000 * 1000:.2f}")
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
            cv2.imwrite('./screenshots/screenshot.png', data)
        else:
            # render camera #1
            self.viewer.render(1920 - 66, 1080 - 55, 0)
            data = np.asarray(self.viewer.read_pixels(1920 - 66, 1080 - 55, depth=False)[::-1, :, :], dtype=np.uint8)
            data[:, :, [0, 2]] = data[:, :, [2, 0]]

            # save data
            if data is not None:
                cv2.imwrite("screenshots/offscreen1.png", data)

            # render camera #2
            self.viewer.render(1920 - 66, 1080 - 55, 1)
            data = np.asarray(self.viewer.read_pixels(1920 - 66, 1080 - 55, depth=False)[::-1, :, :], dtype=np.uint8)
            data[:, :, [0, 2]] = data[:, :, [2, 0]]

            # save data
            if data is not None:
                cv2.imwrite("screenshots/offscreen2.png", data)

    def set_screenshot_flag(self):
        global screenshot_fl
        screenshot_fl = True

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
        if self.toppled_q.empty():
            return False
        else:
            return True

    def exception_occurred(self):
        if self.exception_q.empty():
            return False
        else:
            return True

    def timeout(self):
        if self.timeout_q.empty():
            return False
        else:
            return True

    def pause(self):
        self.pause_fl = True

    def resume(self):
        self.pause_fl = False

    def move_to_block(self, lvl, pos):
        log.debug(f"External move_to_block START!")
        a = Action(ActionType.move_to_block, lvl, pos)
        self.actions_q.put(a)
        while self.simulation_running() and self.state_q.empty():
            time.sleep(0.001)
        if not self.state_q.empty():
            state = self.state_q.get()
        else:
            state = State(None, None, None, None, None, None, None, None, None, Status.over)
        log.debug(f"External move_to_block END!")
        return state

    def _move_to_block(self, level, pos):
        log.debug(f"#1 Move")
        block_id = 3 * level + pos
        self.pusher.move_to_block(block_id)
        force = 0
        block_displacement = 0
        block_positions = self.tower.get_positions()
        tilt = self.tower.get_tilt_2ax(block_positions)
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

    def _push(self):
        start_time = time.time()

        log.debug(f"#1 Push")

        force, block_displacement = self.pusher.push()

        # pause simulation
        self.pause()

        block_positions = self.tower.get_positions()

        start_time_total = time.time()
        # cProfile.run('tilt = self.tower.get_tilt_2ax(block_positions)', 'prof.prof')
        # cProfile.runctx('tilt = self.tower.get_tilt_2ax(block_positions)', globals(), locals(), 'prof.prof')
        start_time = time.time()
        tilt = self.tower.get_tilt_2ax(block_positions)
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

        # resume simulation
        self.resume()
        log.debug(f"#2 Push")
        return state

    def push(self):
        log.debug(f"External push START!")
        a = Action(ActionType.push, 0, 0)
        self.actions_q.put(a)
        while self.simulation_running() and self.state_q.empty():
            time.sleep(0.001)
        if not self.state_q.empty():
            state = self.state_q.get()
        else:
            state = State(None, None, None, None, None, None, None, None, None, Status.over)
        log.debug(f"External push END!")
        return state

    def pull_and_place(self, lvl, pos):
        log.debug(f"External pull_and_place START!")
        a = Action(ActionType.pull_and_place, lvl, pos)
        self.actions_q.put(a)
        while self.simulation_running() and self.state_q.empty():
            time.sleep(0.001)
        if not self.state_q.empty():
            state = self.state_q.get()
        else:
            state = State(None, None, None, None, None, None, None, None, None, Status.over)
        log.debug(f"External pull_and_place END!")
        return state

    def _pull_and_place(self, lvl, pos):
        log.debug(f"#1 Pull")
        id = 3 * lvl + pos
        self.extractor.extract_and_put_on_top(id)
        force = 0
        block_displacement = 0
        block_positions = self.tower.get_positions()
        tilt = self.tower.get_tilt_2ax(block_positions)
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
        return state

    def debug_move_to_zero(self):
        self.extractor.set_position([10, 0, 1])

    def abort_simulation(self):
        self.abort_q.put(1)

    def _abort_simulation(self):
        self.simulation_aborted = True

    def simulate(self):
        log.debug(f"Simulation thread running!")
        # initialize queue for timings
        timings = collections.deque(maxlen=100)

        # initialize viewer
        if self.render:
            if self.on_screen_rendering:
                viewer = MjViewer(self.sim)
            else:
                viewer = MjRenderContextOffscreen(self.sim)

        # start with specific camera position
        if self.render and self.on_screen_rendering:
            viewer.cam.elevation = -37
            viewer.cam.azimuth = 130
            viewer.cam.lookat[0:3] = [-4.12962146, -4.90275717, 7.20812166]
            viewer._run_speed = 128
            viewer.cam.distance = 25

        # initializing step
        # this is needed to initialize positions and orientations of the objects in the scene
        self.sim.step()

        # initialize internal objects
        if self.render:
            self.tower = Tower(self.sim, viewer)
        else:
            self.tower = Tower(self.sim, None)
        self.pusher = Pusher(self.sim, self.tower, self)
        self.extractor = Extractor(self.sim, self.tower, self)

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
                    cycle_elpsed_time = time.time() - cycle_start_time
                    timings.append(cycle_elpsed_time)

                    # check if tower is toppled
                    if self.t % 100 == 0 and self.tower.get_tilt_1ax(self.tower.get_positions()) >= 15:
                        self.toppled_q.put(Error.tower_toppled)
                        log.debug(f"Tower toppled!")

                    # get positions of blocks and plot images
                    if self.t % 100 == 0 and not self.on_screen_rendering:
                        self.off_screen_render_and_plot_errors()

                    # render if on screen rendering
                    if self.render and self.on_screen_rendering:
                        viewer.render()

                    # take a screenshot if q was pressed
                    if self.screenshot_fl:
                        self.screenshot_fl = False
                        self.take_screenshot()

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
            self.exception_q.put(Error.exception_occurred)
        log.debug(f"Exit simulation thread!")

def check_all_blocks(simulation):
    start_time = time.time()
    simulation.move_to_block(0, 0)
    time.sleep(1)
    loose_blocks = []
    force_threshold = 240000
    angle_threshold = 3
    exception_occurred = False
    tower_toppled = False
    timeout = False
    for i in range(g_blocks_num - 9):
        total_displacement = 0
        fl = 0

        if i % 3 == 0:
            force_threshold -= 10500

        for j in range(45):
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

    real_elapsed_time = time.time() - start_time
    # sim_elapsed_time = simulation.t * g_timestep

    return {'extracted_blocks': loose_blocks,
            'real_time': real_elapsed_time,
            # 'sim_time': sim_elapsed_time,
            'error': error,
            }

def run_one_simulation(results, render=True, timeout=600, seed=None):
    log.debug(f"#1 Thread started!")
    env = jenga_env(render=render, timeout=timeout, seed=seed)
    log.debug(f"#2 Thread started!")
    res = check_all_blocks(env)
    log.debug(f"Check stopped!")
    results.append(res)



if __name__ == "__main__":


    start_total_time = time.time()

    N = 8
    TOTAL = 20
    timeout = 600  # in seconds
    render = True
    all_results = []
    counter = N
    threads = []
    seeds = []
    f = open('results.txt', mode='w+')

    for i in range(N):
        seed = time.time_ns()
        t = Process(target=run_one_simulation, args=(all_results, render, timeout, seed))
        t.start()
        threads.append(t)
        seeds.append(seed)

    while counter < TOTAL:
        for i in range(N):
            if not threads[i].is_alive():
                counter += 1
                # create new thread
                seed = time.time_ns()
                t = Process(target=run_one_simulation, args=(all_results, render, timeout, seed))
                t.start()
                threads[i] = t
                seeds.append(seed)

                print(f"All results: {all_results}")
                f.write(f"{all_results[-1]['error']}! Extracted blocks num: {len(all_results[-1]['extracted_blocks'])}. Extracted blocks: {all_results[-1]['extracted_blocks']} Real time: {all_results[-1]['real_time']}. Seed: {seeds[-1]}\n")
                ######################


                while not threads[i].is_alive():
                    time.sleep(1)

        time.sleep(1)

    # wait until all threads are done
    while any(list(map(lambda x: x.is_alive(), threads))):
        time.sleep(1)

    log.debug(f"All results: ")
    for r in all_results:
            log.debug(f"{r['error']}! Extracted blocks num: {len(r['extracted_blocks'])}. Extracted blocks: {r['extracted_blocks']} Real time: {r['real_time']}.")

    elapsed_total_time = time.time() - start_total_time
    f.write(f"Elapsed time in total: {int(elapsed_total_time)}s")
    log.debug(f"Elapsed time in total: {int(elapsed_total_time)}s")


