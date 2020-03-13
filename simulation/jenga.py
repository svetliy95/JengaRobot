from mujoco_py import load_model_from_xml, MjSim, MjViewer, MjRenderContextOffscreen
from pynput import keyboard
import os
from generate_scene import generate_scene
import numpy as np
from pyquaternion import Quaternion
import time
from threading import Thread

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

# specify logger
# DEBUG: Detailed information, typically of interest only when diagnosing problems.
# INFO: Confirmation that things are working as expected.
# WARNING: An indication that something unexpected happened, or indicative of some problem in
# the near future (e.g. ‘disk space low’). The software is still working as expected.
# ERROR: Due to a more serious problem, the software has not been able to perform some function.
# CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
log = logging.Logger(__name__)
# formatter = colorlog.ColoredFormatter('%(log_color)s%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(funcName)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
log.addHandler(stream_handler)

file_logger = logging.Logger("file_logger")
file_formatter = logging.Formatter('%(asctime)s:%(funcName)s:%(message)s')
file_handler = logging.FileHandler(filename='jenga.log')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)
file_logger.addHandler(file_handler)

class Status:
    ready = 'ready'
    ready_to_push = 'ready_to_push'
    pushing = 'pushing'

class jenga_env(gym.Env):

    def __init__(self, render=True):

        # globals
        self.render = render
        self.on_screen_rendering = True
        self.plot_force = False
        self.screenshot_fl = False
        self.automatize_pusher = 1
        self.exception_occurred = False
        self.tower_toppled = False

        # sensor data
        self.g_sensor_data_queue = []
        self.g_sensors_data_queue_maxsize = 250

        # initialize simulation
        self.model = load_model_from_xml(generate_scene(g_blocks_num, g_timestep))
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
            plotting_thread = Thread(target=self.plot_force_data)
            plotting_thread.start()

        # start simulation
        self.simulation_thread = Thread(target=self.simulate)
        self.simulation_thread.start()
        time.sleep(1)

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
                # pusher.move_pusher_in_direction('up')
                # extractor.open_narrow()
                self.extractor.move_in_direction('up')
            if key.char == '-':
                # pusher.move_pusher_in_direction('down')
                # extractor.close_narrow()
                self.extractor.move_in_direction('down')
            if key.char == '.':
                self.pusher.move_pusher_to_next_block()
            if key.char == ',':
                self.pusher.move_pusher_to_previous_block()
            if key.char == 'p':
                # tower.get_tilt_2ax(tower.get_positions())
                # tower.get_tilt_1ax(tower.get_positions())
                log.debug(f'Layers: {self.tower.get_layers(self.tower.get_positions())}')
                log.debug(f"Layers state: {self.tower.get_layers_state(self.tower.get_positions())}")
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
        return not self.exception_occurred

    def terminate_thread(self):
        self.simulation_thread.join()

    def isTowerToppled(self):
        return self.tower_toppled

    def pause(self):
        self.pause_fl = True

    def resume(self):
        self.pause_fl = False

    def move_to_block(self, level, pos):
        block_id = 3 * level + pos
        self.pusher.move_to_block(block_id)

    def push(self):

        log.debug(f"Push")
        force, block_displacement = self.pusher.push()
        block_positions = self.tower.get_positions()
        tilt = self.tower.get_tilt_2ax(block_positions)
        displacement_total = self.tower.get_abs_displacement_2ax(self.pusher.current_block, block_positions)
        last_displacement = self.tower.get_last_displacement_2ax(self.pusher.current_block, block_positions)
        z_rotation_total = self.tower.get_total_z_rotation(self.pusher.current_block)
        z_rotation_last = self.tower.get_last_z_rotation(self.pusher.current_block)
        status = Status.pushing
        self.total_distance += block_displacement
        start_time = time.time()
        tower_state = self.tower.get_layers_state(block_positions)
        elapsed_time = time.time() - start_time
        log.debug(f"Layers: {tower_state}")

        state = {'layers': tower_state,
                 'force': force,
                 'block_displacement': block_displacement,
                 'total_block_displacement': self.total_distance,
                 'tilt': tilt,
                 'mean_displacement_total': displacement_total,
                 'mean_displacement_last': last_displacement,
                 'z_rotation_total': z_rotation_total,
                 'z_rotation_last': z_rotation_last,
                 'status': status
                 }


        log.debug(f"Push elapsed: {elapsed_time}")
        return state

    def pull_and_place(self, id):
        self.extractor.extract_and_put_on_top(id)

    def simulate(self):
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
            viewer._run_speed = 1
            viewer.cam.distance = 25

        # initialize internal objects
        if self.render:
            self.tower = Tower(self.sim, viewer)
        else:
            self.tower = Tower(self.sim, None)
        self.pusher = Pusher(self.sim, self.tower)
        self.extractor = Extractor(self.sim, self.tower)

        # simulation loop
        try:
            while not self.tower_toppled:
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
                        self.tower_toppled = True

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


                    log.debug(f"Cycle time: {np.mean(timings)*1000}ms")
                    if self.t > 100 and os.getenv('TESTING') is not None:
                        break
                else:
                    time.sleep(0.1)
        except Exception:
            self.exception_occurred = True

def check_all_blocks(simulation):
    simulation.move_to_block(0, 0)
    time.sleep(1)
    loose_blocks = []
    force_threshold = 240000
    angle_threshold = 3
    simulation_over = False
    tower_toppled = False
    for i in range(g_blocks_num - 9):
        total_displacement = 0
        fl = 0

        if i % 3 == 0:
            force_threshold -= 10500

        for j in range(45):
            if (not simulation.simulation_running) or simulation.isTowerToppled():
                simulation_over = not simulation.simulation_running()
                tower_toppled = simulation.isTowerToppled()
                fl = 1
                break
            force, displacement = simulation.pusher.push()
            tilt = simulation.tower.get_tilt_1ax(simulation.tower.get_positions())
            total_displacement += displacement
            if force > force_threshold or tilt > angle_threshold:
                fl = 1
                break
        if fl == 0:
            loose_blocks.append(i)
            simulation.pull_and_place(i)

        if simulation_over:
            log.debug(f"Simulation became unstable")
            break
        if tower_toppled:
            log.debug(f"Tower toppled!")
            break

        simulation.move_to_block((i + 1) // 3, (i + 1) % 3)
        time.sleep(1)
    print(loose_blocks)

if __name__ == "__main__":
        simulation = jenga_env(render=True)
        check_all_blocks(simulation)
        time.sleep(1)
