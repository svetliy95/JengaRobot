from mujoco_py import load_model_from_xml, MjSim, MjViewer
from pynput import keyboard
import os
from generate_scene import generate_scene, block_length_mean, pusher_size, one_millimeter, block_sizes
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
from pyquaternion import Quaternion
import time
import threading
from multiprocessing import Process

# animated plot
import pyformulas as pf
import queue
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


# globals
g_blocks_num = 54
g_x_pusher = 3
g_y_pusher = 3
g_z_pusher = 3
g_quat_pusher = Quaternion([1, 0, 0, 0])
g_current_block_for_pusher = 0
g_steps_for_pusher = 0
g_sensor_data_queue = []
g_sensors_data_queue_maxsize = 250


def plot_force_data():
    fig = plt.figure()
    canvas = np.zeros((48, 64))
    screen = pf.screen(canvas, 'Force values')

    while True:
        fig.clear()
        ys = g_sensor_data_queue  # necessary because of no thread synchronisation
        xs = range(-len(ys) + 1, 1)
        plt.plot(xs, ys, c='black')

        start = time.time()
        fig.canvas.draw()
        stop = time.time()
        print(stop - start)
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        screen.update(image)


def update_force_sensor_plot():
    current_sensor_value = sim.data.sensordata[g_blocks_num*6]
    if len(g_sensor_data_queue) >= g_sensors_data_queue_maxsize:
        g_sensor_data_queue.pop(0)
    g_sensor_data_queue.append(current_sensor_value)


def push_forward():
    global g_steps_for_pusher, g_x_pusher, g_y_pusher, g_z_pusher
    x_unit_vector = [1, 0, 0]
    if g_steps_for_pusher > 0:
        translation = np.array(g_quat_pusher.rotate(x_unit_vector)) * one_millimeter
        g_x_pusher -= translation[0]
        g_y_pusher -= translation[1]
        g_z_pusher -= translation[2]
        g_steps_for_pusher -= 1


def point_projection_on_line(line_point1, line_point2, point):
    print(line_point1)
    print(line_point2)
    print(point)
    ap = point - line_point1
    ab = line_point2 - line_point1
    print()
    result = line_point1 + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result


def move_pusher_to_next_block():
    global g_current_block_for_pusher
    if g_current_block_for_pusher < g_blocks_num - 1:
        g_current_block_for_pusher += 1
        move_pusher_to_block(g_current_block_for_pusher)


def move_pusher_to_previous_block():
    global g_current_block_for_pusher
    if g_current_block_for_pusher > 0:
        g_current_block_for_pusher -= 1
        move_pusher_to_block(g_current_block_for_pusher)


def move_pusher_with_arrows(direction):  # 'left', 'right', 'forward', 'backwards', 'up', 'down'
    global g_quat_pusher, g_x_pusher, g_y_pusher, g_z_pusher
    x_unit_vector = [1, 0, 0]
    y_unit_vector = [0, 1, 0]
    z_unit_vector = [0, 0, 1]
    if(direction == 'left'):
        translation = np.array(g_quat_pusher.rotate(y_unit_vector)) * one_millimeter
        g_x_pusher -= translation[0]
        g_y_pusher -= translation[1]
        g_z_pusher -= translation[2]
    if(direction == 'right'):
        translation = np.array(g_quat_pusher.rotate(y_unit_vector)) * one_millimeter
        g_x_pusher += translation[0]
        g_y_pusher += translation[1]
        g_z_pusher += translation[2]
    if(direction == 'forward'):
        translation = np.array(g_quat_pusher.rotate(x_unit_vector)) * one_millimeter
        g_x_pusher -= translation[0]
        g_y_pusher -= translation[1]
        g_z_pusher -= translation[2]
    if(direction == 'backwards'):
        translation = np.array(g_quat_pusher.rotate(x_unit_vector)) * one_millimeter
        g_x_pusher += translation[0]
        g_y_pusher += translation[1]
        g_z_pusher += translation[2]
    if(direction == 'up'):
        translation = np.array(g_quat_pusher.rotate(z_unit_vector)) * one_millimeter
        g_x_pusher += translation[0]
        g_y_pusher += translation[1]
        g_z_pusher += translation[2]
    if(direction == 'down'):
        translation = np.array(g_quat_pusher.rotate(z_unit_vector)) * one_millimeter
        g_x_pusher -= translation[0]
        g_y_pusher -= translation[1]
        g_z_pusher -= translation[2]


def move_pusher_to_block(block_num):
    global g_x_pusher
    global g_y_pusher
    global g_z_pusher
    global g_quat_pusher
    x_unit_vector = np.array([2, 0, 0])
    y_unit_vector = np.array([0, 2, 0])

    # get orientation of the target block as a quaternion
    block_quat = Quaternion(sim.data.sensordata[3*54 + block_num*4 + 0],
                   sim.data.sensordata[3*54 + block_num*4 + 1],
                   sim.data.sensordata[3*54 + block_num*4 + 2],
                   sim.data.sensordata[3*54 + block_num*4 + 3])
    block_x_face_normal_vector = block_quat.rotate(x_unit_vector)
    target_x = sim.data.sensordata[block_num * 3 + 0] - 3 + (block_sizes[block_num][0] / 2 + 2 * pusher_size) * block_x_face_normal_vector[0]
    target_y = sim.data.sensordata[block_num * 3 + 1] - 3 + (block_sizes[block_num][0] / 2 + 2 * pusher_size) * block_x_face_normal_vector[1]
    target_z = sim.data.sensordata[block_num * 3 + 2] - 3 + (block_sizes[block_num][0] / 2 + 2 * pusher_size) * block_x_face_normal_vector[2]

    # move pusher along its own y axis first
    translation_direction = g_quat_pusher.rotate(y_unit_vector)
    stopover = point_projection_on_line(np.array([g_x_pusher, g_y_pusher, g_z_pusher]),
                                        np.array([g_x_pusher, g_y_pusher, g_z_pusher]) + translation_direction,
                                        np.array([target_x, target_y, target_z]))
    g_x_pusher = stopover[0]
    g_y_pusher = stopover[1]
    g_z_pusher = stopover[2]
    time.sleep(0.2)
    g_quat_pusher = block_quat
    g_x_pusher = target_x
    g_y_pusher = target_y
    g_z_pusher = target_z


def set_pusher_pos(sim, x, y, z, quat):
    yaw_pitch_roll = quat.yaw_pitch_roll
    sim.data.ctrl[0] = x
    sim.data.ctrl[1] = y
    sim.data.ctrl[2] = z
    sim.data.ctrl[3] = yaw_pitch_roll[2]
    sim.data.ctrl[4] = yaw_pitch_roll[1]
    sim.data.ctrl[5] = yaw_pitch_roll[0]


def on_press(key):
    global g_x_pusher
    global g_y_pusher
    global g_steps_for_pusher

    try:
        if key.char == '5':
            threading.Thread(target=move_pusher_to_block, args=(5,)).start()
        if key.char == '6':
            threading.Thread(target=move_pusher_to_block, args=(6,)).start()
        if key.char == '7':
            threading.Thread(target=move_pusher_to_block, args=(7,)).start()
        if key.char == '8':
            threading.Thread(target=move_pusher_to_block, args=(8,)).start()
        if key.char == '9':
            threading.Thread(target=move_pusher_to_block, args=(9,)).start()
        if key.char == '+':
            move_pusher_with_arrows('up')
        if key.char == '-':
            move_pusher_with_arrows('down')
        if key.char == '.':
            move_pusher_to_next_block()
        if key.char == ',':
            move_pusher_to_previous_block()
        if key.char == 'p':
            g_steps_for_pusher = 45
            print("45")
    except AttributeError:
        if key == keyboard.Key.up:
            move_pusher_with_arrows('forward')
        if key == keyboard.Key.down:
            move_pusher_with_arrows('backwards')
        if key == keyboard.Key.left:
            move_pusher_with_arrows('left')
        if key == keyboard.Key.right:
            move_pusher_with_arrows('right')
    print(key)


def start_keyboard_listener():
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()


model = load_model_from_xml(generate_scene(g_blocks_num))
sim = MjSim(model)
viewer = MjViewer(sim)
# start with specific camera position
viewer.cam.elevation = -7
viewer.cam.azimuth = 180
viewer._run_speed = 1
viewer.cam.lookat[2] = -1
viewer.cam.distance = 15
viewer._run_speed = 1
t = 0

# initial position of pusher
move_pusher_to_block(g_current_block_for_pusher)

# start keyboard listener
start_keyboard_listener()

threading.Thread(target=plot_force_data).start()
# fig = plt.figure()
# canvas = np.zeros((48, 64))
# screen = pf.screen(canvas, 'Force values')

while True:
    t += 1


    push_forward()
    set_pusher_pos(sim, g_x_pusher, g_y_pusher, g_z_pusher, g_quat_pusher)
    sim.step()


    update_force_sensor_plot()



    # fig.clear()

    # plt.plot(xs, g_sensor_data_queue, c='black')

    start = time.time()
    # fig.canvas.draw()
    # stop = time.time()
    # print(stop - start)
    # image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # screen.update(image)



    viewer.render()



    if t > 100 and os.getenv('TESTING') is not None:
        break

