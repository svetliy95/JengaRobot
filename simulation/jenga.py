from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os
from generate_scene import generate_scene
import time
import tensorflow

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

while True:
    # sim.data.ctrl[0] = math.cos(t / 1000.) # * 0.01
    # sim.data.ctrl[1] = math.sin(t / 1000.) # * 0.01

    t += 1

    start = time.time()
    sim.step()
    viewer.render()
    stop = time.time()

    print(sim.data.body_xpos[4])
    print(sim.data.body_xquat[4])
    print("\n")
    print(stop - start)

    if t > 100 and os.getenv('TESTING') is not None:
        break
