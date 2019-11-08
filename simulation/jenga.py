from mujoco_py import load_model_from_xml, MjSim, MjViewer
from pynput import keyboard
import os
from generate_scene import generate_scene, block_length_mean, pusher_size, one_millimeter
import time
from scipy.spatial.transform import Rotation as R
import threading

x_pusher = 0
y_pusher = 0
z_pusher = 0


def set_pusher_pos(sim, x, y, z):
    sim.data.ctrl[0] = x
    sim.data.ctrl[1] = y
    sim.data.ctrl[2] = z


def on_press(key):
    if key == keyboard._xorg.Key.up:
        print("hi!")
        x_pusher += one_millimeter


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
x_pusher = sim.data.sensordata[0] - 3 + block_length_mean/2 + pusher_size
y_pusher = sim.data.sensordata[1] - 3
z_pusher = sim.data.sensordata[2] - 3

# start keyboard listener
listener_thread = threading.Thread(target=start_keyboard_listener)
listener_thread.start()

while True:
    t += 1

    start = time.time()
    sim.data.ctrl[0] = x_pusher
    sim.data.ctrl[1] = y_pusher
    sim.data.ctrl[2] = z_pusher
    set_pusher_pos(sim, sim.data.sensordata[0] - 3 + block_length_mean/2 + pusher_size, sim.data.sensordata[1] - 3, sim.data.sensordata[2] - 3)
    sim.step()
    viewer.render()
    stop = time.time()

    r = R.from_quat(sim.data.body_xquat[4])
    # print(r.as_euler('xyz', degrees=True))


    # print(stop - start)
    # print("\n")


    if t > 100 and os.getenv('TESTING') is not None:
        break

