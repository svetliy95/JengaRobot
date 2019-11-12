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

g_x_pusher = 0
g_y_pusher = 0
g_z_pusher = 0


def move_pusher_to_block(block_num):
    global g_x_pusher
    global g_y_pusher
    global g_z_pusher
    x_unit_vector = np.array([5, 0, 0])

    # read quaternion
    q = Quaternion(sim.data.sensordata[3*54 + block_num*4 + 0],
                   sim.data.sensordata[3*54 + block_num*4 + 1],
                   sim.data.sensordata[3*54 + block_num*4 + 2],
                   sim.data.sensordata[3*54 + block_num*4 + 3])
    rotated_vector = q.rotate(x_unit_vector)

    g_x_pusher = sim.data.sensordata[block_num * 3 + 0] - 3 + (block_sizes[block_num][0] / 2 + 2 * pusher_size) * rotated_vector[0]
    time.sleep(0.5)
    g_y_pusher = sim.data.sensordata[block_num * 3 + 1] - 3 + (block_sizes[block_num][0] / 2 + 2 * pusher_size) * rotated_vector[1]
    g_z_pusher = sim.data.sensordata[block_num * 3 + 2] - 3 + (block_sizes[block_num][0] / 2 + 2 * pusher_size) * rotated_vector[2]


def set_pusher_pos(sim, x, y, z):
    sim.data.ctrl[0] = x
    sim.data.ctrl[1] = y
    sim.data.ctrl[2] = z


def on_press(key):
    global g_x_pusher
    global g_y_pusher

    try:
        if key.char == '5':
            threading.Thread(target=move_pusher_to_block, args=(5,)).start()
        if key.char == '6':
            move_pusher_to_block(6)
        if key.char == '7':
            move_pusher_to_block(7)
        if key.char == '8':
            move_pusher_to_block(8)
        if key.char == '9':
            move_pusher_to_block(9)
    except AttributeError:
        if key == keyboard.Key.up:
            g_x_pusher -= one_millimeter
        if key == keyboard.Key.down:
            g_x_pusher += one_millimeter
        if key == keyboard.Key.left:
            g_y_pusher -= one_millimeter
        if key == keyboard.Key.right:
            g_y_pusher += one_millimeter

    print(key)



def start_keyboard_listener():
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()


model = load_model_from_xml(generate_scene())
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
g_x_pusher = sim.data.sensordata[0] - 3 + block_length_mean / 2 + 2 * pusher_size
g_y_pusher = sim.data.sensordata[1] - 3
g_z_pusher = sim.data.sensordata[2] - 3

# start keyboard listener
start_keyboard_listener()

while True:
    t += 1

    start = time.time()
    set_pusher_pos(sim, g_x_pusher, g_y_pusher, g_z_pusher)
    # sim.data.ctrl[0] = g_x_pusher
    # sim.data.ctrl[1] = g_y_pusher
    # sim.data.ctrl[2] = g_z_pusher
    sim.step()
    viewer.render()
    stop = time.time()

    r = R.from_quat(sim.data.body_xquat[4])
    # print(r.as_euler('xyz', degrees=True))


    # print(stop - start)
    # print("\n")


    if t > 100 and os.getenv('TESTING') is not None:
        break

