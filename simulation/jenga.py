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


class jenga_env(gym.Env):

    def __init__(self):
        # specify logger
        # DEBUG: Detailed information, typically of interest only when diagnosing problems.
        # INFO: Confirmation that things are working as expected.
        # WARNING: An indication that something unexpected happened, or indicative of some problem in
        # the near future (e.g. ‘disk space low’). The software is still working as expected.
        # ERROR: Due to a more serious problem, the software has not been able to perform some function.
        # CRITICAL: A serious error, indicating that the program itself may be unable to continue running.
        self.log = logging.Logger(__name__)
        # formatter = colorlog.ColoredFormatter('%(log_color)s%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
        self.formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(funcName)s:%(message)s')
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setFormatter(self.formatter)
        self.stream_handler.setLevel(logging.DEBUG)
        self.log.addHandler(self.stream_handler)

        # globals
        self.on_screen_rendering = True
        self.plot_force = False
        self.screenshot_fl = False
        self.automatize_pusher = 0

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
        Thread(target=self.simulate).start()
        self.sleep_simtime(0.5)
        self.pause()

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
                self.get_camera_pose()
                self.log.debug(f'Total z rotation: {self.tower.get_total_z_rotation(0)}')
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

    def check_all_blocks(self):
        self.pusher.move_to_block(0)
        time.sleep(1)
        loose_blocks = []
        force_threshold = 240000
        angle_threshold = 3
        for i in range(g_blocks_num - 3):
            total_displacement = 0
            fl = 0

            if i % 3 == 0:
                force_threshold -= 10500

            for j in range(45):
                force, displacement = self.pusher.push()
                total_displacement += displacement
                self.log.debug(f"Total displacement: {self.tower.get_abs_displacement_2ax(i, self.tower.get_positions()) / one_millimeter}")
                self.log.debug(f"Last displacement: {self.tower.get_last_displacement_2ax(i, self.tower.get_positions()) / one_millimeter}")
                if force > force_threshold or self.tower.get_angle_of_highest_block_to_ground(self.tower.get_positions()) > angle_threshold:
                    fl = 1
                    break
            if fl == 0:
                loose_blocks.append(i)
                self.log.debug(f"Loose blocks: {loose_blocks}")
                self.log.debug(f"Highest blocks: {self.tower.get_blocks_from_highest_level(self.tower.get_positions())}")
                self.log.debug(self.tower.get_full_layers(self.tower.get_positions()))
                self.log.debug(self.tower.get_center_xy(self.tower.get_positions()))
                self.extractor.extract_and_put_on_top(i)
            self.pusher.move_pusher_to_next_block()
            time.sleep(1)
        print(loose_blocks)

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

    def pause(self):
        self.pause_fl = True

    def resume(self):
        self.pause_fl = False

    def move_to_block(self, level, pos):
        block_id = 3 * level + pos
        self.pusher.move_to_block(block_id)

    def push(self):
        self.pusher.push()

    def simulate(self):
        # initialize viewer
        if self.on_screen_rendering:
            viewer = MjViewer(self.sim)
        else:
            viewer = MjRenderContextOffscreen(self.sim)

        # start with specific camera position
        if self.on_screen_rendering:
            viewer.cam.elevation = -37
            viewer.cam.azimuth = 130
            viewer.cam.lookat[0:3] = [-4.12962146, -4.90275717, 7.20812166]
            viewer._run_speed = 1
            viewer.cam.distance = 25

        # initialize internal objects
        self.tower = Tower(self.sim, viewer)
        self.pusher = Pusher(self.sim, self.tower)
        self.extractor = Extractor(self.sim, self.tower)

        # simulation loop
        while True:
            if not self.pause:
                self.t += 1
                cycle_start_time = time.time()

                # update positions of the pusher and gripper
                self.pusher.update_position(self.t)
                self.extractor.update_positions(self.t)

                # make step
                self.sim.step()

                # get positions of blocks and plot images
                if self.t % 100 == 0 and not self.on_screen_rendering:
                    self.off_screen_render_and_plot_errors()

                # render if on screen rendering
                if self.on_screen_rendering:
                    viewer.render()

                # take a screenshot if q was pressed
                if self.screenshot_fl:
                    self.screenshot_fl = False
                    self.take_screenshot()

                # update force sensor data if enabled
                if self.plot_force:
                    self.update_force_sensor_plot()

                cycle_elpsed_time = time.time() - cycle_start_time
                if self.t > 100 and os.getenv('TESTING') is not None:
                    break
            else:
                time.sleep(0.1)

if __name__ == "__main__":
    simulation = jenga_env()
