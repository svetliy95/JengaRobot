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
from math import sin, cos, radians
from cv.transformations import matrix2pose
import cv2

import math


# globals
g_sensor_data_queue = []
g_sensors_data_queue_maxsize = 250
screenshot_fl = False

# floating body
fb_x = -7
fb_y = 0
fb_z = 3
fb_yaw = 0
fb_pitch = 0
fb_roll = 0


def print_fixed_camera_xml(cam_pos, cam_lookat):
    cam_zaxis = cam_pos - cam_lookat
    plane_normal = np.array([0, 0, 1])
    cam_xaxis = -np.cross(cam_zaxis, plane_normal)
    cam_yaxis = np.cross(cam_zaxis, cam_xaxis)

    print(f'pos="{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}"'
          f' xyaxes="{cam_xaxis[0]} {cam_xaxis[1]} {cam_xaxis[2]} {cam_yaxis[0]} {cam_yaxis[1]} {cam_yaxis[2]}"')


def get_camera_pose():
    # TODO: return orientation
    elevation = radians(-viewer.cam.elevation)
    azimuth = radians(viewer.cam.azimuth)
    lookat = np.array(viewer.cam.lookat)
    distance = viewer.cam.distance

    z = lookat[2] + sin(elevation) * distance
    proj_dist = cos(elevation) * distance

    x = lookat[0] - proj_dist * cos(azimuth)
    y = lookat[1] - proj_dist * sin(azimuth)

    return np.array([x, y, z, 0, 0, 0])


def plot_force_data():
    fig = plt.figure()
    canvas = np.zeros((48, 64))
    screen = pf.screen(canvas, 'Force values')

    while True:
        fig.clear()
        ys = -np.array(g_sensor_data_queue)  # copy data because of no thread synchronisation
        xs = range(-len(ys) + 1, 1)
        plt.plot(xs, ys, c='black')
        fig.canvas.draw()
        stop = time.time()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        screen.update(image)


def update_force_sensor_plot():
    current_sensor_value = sim.data.sensordata[g_blocks_num*3 + g_blocks_num*4]
    if len(g_sensor_data_queue) >= g_sensors_data_queue_maxsize:
        g_sensor_data_queue.pop(0)
    g_sensor_data_queue.append(current_sensor_value)


def start_keyboard_listener():
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()


def on_press(key):
    global fb_x, fb_y, fb_z, fb_pitch, fb_yaw, fb_roll

    try:
        if key.char == '5':
            Thread(target=pusher.move_pusher_to_block, args=(5,)).start()
        if key.char == '6':
            Thread(target=pusher.move_pusher_to_block, args=(6,)).start()
        if key.char == '7':
            Thread(target=pusher.move_pusher_to_block, args=(7,)).start()
        if key.char == '8':
            Thread(target=pusher.move_pusher_to_block, args=(8,)).start()
        if key.char == '9':
            Thread(target=pusher.move_pusher_to_block, args=(9,)).start()
        if key.char == '+':
            # pusher.move_pusher_in_direction('up')
            fb_z += 0.1
        if key.char == '-':
            # pusher.move_pusher_in_direction('down')
            fb_z -= 0.1
        if key.char == '.':
            pusher.move_pusher_to_next_block()
        if key.char == ',':
            pusher.move_pusher_to_previous_block()
        if key.char == 'p':
            pusher.push()
        if key.char == 'q':
            take_screenshot()
    except AttributeError:
        if key == keyboard.Key.up:
            # pusher.move_pusher_in_direction('forward')
            fb_x -= 0.1
        if key == keyboard.Key.down:
            # pusher.move_pusher_in_direction('backwards')
            fb_x += 0.1
        if key == keyboard.Key.left:
            # pusher.move_pusher_in_direction('left')
            fb_y -= 0.1
        if key == keyboard.Key.right:
            # pusher.move_pusher_in_direction('right')
            fb_y += 0.1
    print(key)


def check_all_blocks():
    pusher.move_pusher_to_block(0)
    time.sleep(1)
    loose_blocks = []
    force_threshold = 240000
    angle_threshold = 3
    for i in range(g_blocks_num - 3):
        total_displacement = 0
        fl = 0

        if i % 3 == 0:
            force_threshold -= 10500

        for j in range(50):
            force, displacement = pusher.push()
            total_displacement += displacement
            if force > force_threshold or tower.get_angle_of_highest_block_to_ground() > angle_threshold:
                fl = 1
                break
        if fl == 0:
            loose_blocks.append(i)
        pusher.move_pusher_to_next_block()
        time.sleep(1)
    print(loose_blocks)


def take_screenshot():
    global screenshot_fl
    screenshot_fl = True


# initialize simulation
on_screen_rendering = False
plot_force = False
automatize_pusher = True
model = load_model_from_xml(generate_scene(g_blocks_num, g_timestep))
sim = MjSim(model)
if on_screen_rendering:
    viewer = MjViewer(sim)
else:
    viewer = MjRenderContextOffscreen(sim)

pusher = Pusher(sim)
tower = Tower(sim, viewer)

# start keyboard listener
start_keyboard_listener()

# start with specific camera position
if on_screen_rendering:
    viewer.cam.elevation = -7
    viewer.cam.azimuth = 180
    viewer._run_speed = 1
    viewer.cam.lookat[2] = -1
    viewer.cam.distance = 15




# start force data plotting
if plot_force:
    plotting_thread = Thread(target=plot_force_data)
    plotting_thread.start()

# create histogram plot
x_errors = np.arange(g_blocks_num)
y_errors = np.zeros(g_blocks_num)
plt.ion()
fig_errors = plt.figure()
ax = fig_errors.add_subplot(111)
ax.set(ylim=(0, 370))
ax.xaxis.set_ticks(range(g_blocks_num))
ax.yaxis.set_ticks(range(10))
line1 = ax.bar(x_errors, y_errors)
ax.grid(zorder=0)

# simulation loop
t = 0
while True:
    t += 1

    cycle_time_start = time.time()

    pusher.update_position(t)
    start = time.time()
    sim.step()
    stop = time.time()

    print(f"Step time: {stop-start}")

    # get positions of blocks and plot images
    if t % 100 == 0:

        positions, im1, im2 = tower.get_pose_cv(range(g_blocks_num))

        # get actual and estimated positions
        actual_positions = []
        estimated_positions = []
        used_tags = []
        for i in range(g_blocks_num):
            actual_positions.append(tower.get_position(i))
        actual_positions = np.array(actual_positions)

        # extract estimated positions
        h, w, _ = im1.shape
        new_h = int(1080 / 2.2)
        new_w = int(1920 / 2.2)
        im1 = cv2.resize(im1, (new_w, new_h))
        im2 = cv2.resize(im2, (new_w, new_h))
        print(im1.shape)
        black_line = np.zeros((new_h, 10, 3), np.uint8)
        print(black_line.shape)
        im = np.concatenate((im1, black_line, im2), axis=1)
        cv2.imshow(mat=im, winname="Render")
        start = time.time()
        cv2.waitKey(1)
        stop = time.time()
        print(f"Wait_time: {stop - start} s.")
        for i in range(g_blocks_num):
            est_pose = positions.get(i)
            if est_pose is not None:
                pos_error = est_pose['orientation_error'] / one_millimeter
                line1[i].set_height(pos_error)
                if est_pose['tags_detected'] == 2:
                    line1[i].set_color('b')
                else:
                    line1[i].set_color('y')
            else:
                line1[i].set_height(10)
                line1[i].set_color('r')

        for i in range(g_blocks_num):
            print(f"CV: #{i}: {positions[i]['orientation']}")
            print(f"Ac: #{i}: {tower.get_orientation(i)}")


        fig_errors.canvas.draw()
        fig_errors.canvas.flush_events()

    if on_screen_rendering:
        viewer.render()

    if screenshot_fl:
        screenshot_fl = False
        if on_screen_rendering:
            data = np.asarray(viewer.read_pixels(1920 - 66, 1080 - 55, depth=False)[::-1, :, :], dtype=np.uint8)
            data[:, :,  [0, 2]] = data[:, :, [2, 0]]
            cv2.imwrite('./screenshots/screenshot.png', data)
        else:
            # render camera #1
            viewer.render(1920-66, 1080-55, 0)
            data = np.asarray(viewer.read_pixels(1920-66, 1080-55, depth=False)[::-1, :, :], dtype=np.uint8)
            data[:, :, [0, 2]] = data[:, :, [2, 0]]

            # save data
            if data is not None:
                cv2.imwrite("screenshots/offscreen1.png", data)

            # render camera #2
            viewer.render(1920 - 66, 1080 - 55, 1)
            data = np.asarray(viewer.read_pixels(1920 - 66, 1080 - 55, depth=False)[::-1, :, :], dtype=np.uint8)
            data[:, :, [0, 2]] = data[:, :, [2, 0]]

            # save data
            if data is not None:
                cv2.imwrite("screenshots/offscreen2.png", data)

    if plot_force:
        update_force_sensor_plot()
    if t == 100:
        # start block checking
        if automatize_pusher:
            checking_thread = Thread(target=check_all_blocks)
            checking_thread.start()

    # print_fixed_camera_xml(get_camera_pose()[:3], viewer.cam.lookat)

    # debug outputs
    q = Quaternion([1, 0, 0, 0])
    q = Quaternion(axis=[0.4, 1, -0.6], angle=(math.pi/2)*2.5)
    # q = Quaternion([-0.697, -0.001, 0.717, -0.006])
    # q = Quaternion([-0.704, 0.711, 0.002, 0.009])
    q = Quaternion([-0.704, 0.71, 0.002, 0.009])
    # print(q.elements)
    ypr = q.yaw_pitch_roll
    # sim.data.set_mocap_pos("floating_body", [fb_x, fb_y, fb_z])
    # sim.data.set_mocap_quat("floating_body", q.elements)

    # print(np.array_str(tower.get_orientation(3), precision=3, suppress_small=True))
    # print(np.array_str(tower.get_position(3)/one_millimeter, precision=3, suppress_small=True))

    # print mocap body position
    # print(np.array_str((sim.data.body_xpos[-1] + np.array([-0.03*scaler*0.8, -0.03*scaler*0.8, +0.03*scaler]))/one_millimeter, precision=3, suppress_small=True))
    # print(np.array_str(sim.data.body_xquat[-1], precision=3, suppress_small=True))


    # calculate mean penetration
    # penetrations = []
    # for x in sim.data.contact:
    #     if x.dist != 0.0:
    #         penetrations.append(x.dist)
    #
    # print(np.mean(penetrations))




    cycle_time_stop = time.time()
    print(f"Cycle time: {cycle_time_stop - cycle_time_start} s.")
    if t > 100 and os.getenv('TESTING') is not None:
        break