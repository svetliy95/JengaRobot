from mujoco_py import load_model_from_xml, MjSim, MjViewer
from pynput import keyboard
import os
from generate_scene import generate_scene
import time
from scipy.spatial.transform import Rotation as R
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


# globals
g_sensor_data_queue = []
g_sensors_data_queue_maxsize = 250


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
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        screen.update(image)


def update_force_sensor_plot():
    current_sensor_value = sim.data.sensordata[g_blocks_num*3 + g_blocks_num*4]
    if len(g_sensor_data_queue) >= g_sensors_data_queue_maxsize:
        g_sensor_data_queue.pop(0)
    g_sensor_data_queue.append(current_sensor_value)


def on_press(key):

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
            pusher.move_pusher_in_direction('up')
        if key.char == '-':
            pusher.move_pusher_in_direction('down')
        if key.char == '.':
            pusher.move_pusher_to_next_block()
        if key.char == ',':
            pusher.move_pusher_to_previous_block()
        if key.char == 'p':
            pusher.push()
    except AttributeError:
        if key == keyboard.Key.up:
            pusher.move_pusher_in_direction('forward')
        if key == keyboard.Key.down:
            pusher.move_pusher_in_direction('backwards')
        if key == keyboard.Key.left:
            pusher.move_pusher_in_direction('left')
        if key == keyboard.Key.right:
            pusher.move_pusher_in_direction('right')
    print(key)


def check_all_blocks():
    loose_blocks = []
    for i in range(g_blocks_num - 3):
        total_displacement = 0
        fl = 0
        for j in range(50):
            force, displacement = pusher.push()
            total_displacement += displacement
            if force > force_threshold:
                fl = 1
                break
        if fl == 0:
            loose_blocks.append(i)
        pusher.move_pusher_to_next_block()
        time.sleep(1)
    print(loose_blocks)






def start_keyboard_listener():
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()


model = load_model_from_xml(generate_scene(g_blocks_num, g_timestep))
sim = MjSim(model)
viewer = MjViewer(sim)
pusher = Pusher(sim)

# start with specific camera position
viewer.cam.elevation = -7
viewer.cam.azimuth = 180
viewer._run_speed = 1
viewer.cam.lookat[2] = -1
viewer.cam.distance = 15
viewer._run_speed = 1
t = 0

# initial position of pusher
pusher.move_pusher_to_block(0)

# start keyboard listener
start_keyboard_listener()

# start force data plotting
plotting_thread = Thread(target=plot_force_data)
plotting_thread.start()



while True:
    t += 1


    # push_forward()
    pusher.update_position(t)
    start = time.time()
    sim.step()
    stop = time.time()
    # if(t % 10 == 0):
    #     print(g_timestep/(stop - start))

    update_force_sensor_plot()
    viewer.render()

    if t == 100:
        # start block checking
        checking_thread = Thread(target=check_all_blocks)
        checking_thread.start()



    if t > 100 and os.getenv('TESTING') is not None:
        break
plotting_thread.join()