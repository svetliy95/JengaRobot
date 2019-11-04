from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>

    <statistic extent="2" meansize=".05"/>

    <option timestep="0.0005" solver="CG" integrator="RK4" iterations="100" tolerance="1e-10" jacobian="sparse" cone="pyramidal"/>

    <visual>
        <rgba haze="0.15 0.25 0.35 1"/>
        <quality shadowsize="2048"/>
        <map stiffness="700" shadowscale="0.5" fogstart="10" fogend="15" zfar="40" haze="0.3"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/> 
        <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" 
            width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
        <texture name="texsponge" type="2d" file="/home/bch_svt/cartpole/simulation/sponge.png"/>   
        <texture name="texcarpet" type="2d" file="/home/bch_svt/cartpole/simulation/carpet.png"/>  
        
        <texture name="texmarble" type="cube" file="/home/bch_svt/cartpole/simulation/marble.png"/>s

        <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
        <material name="matcarpet" texture="texcarpet"/>
        <material name="matsponge" texture="texsponge" specular="0.3"/>
        <material name="matmarble" texture="texmarble" rgba=".7 .7 .7 1"/>
    </asset>
    
    <worldbody>
        <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1" castshadow="false"/>
        <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>
        <geom name="ground" type="plane" size="0 0 1" pos="0 0 -1" quat="1 0 0 0" material="matplane" condim="1"/>
        
        <body name="block1" pos="0.0 0.051 0.015">
            <freejoint/>
            <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
			<camera euler="0 0 0" fovy="40" name="rgb" pos="0 0 0.025"></camera>
        </body>
        <body name="block2" pos="0 0 0.015">
            <freejoint/>
            <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
        </body>
        <body name="block3" pos="0.0 -0.051 0.015">
            <freejoint/>
            <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
        </body>
        
        <body name="block4" pos="0.051 0 0.045" euler="0 0 90">
            <freejoint/>
            <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
        </body>
        <body name="block5" pos="0 0 0.045" euler="0 0 90">
            <freejoint/>
            <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
        </body>
        <body name="block6" pos="-0.051 0 0.045" euler="0 0 90">
            <freejoint/>
            <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
        </body>
        
        
    </worldbody>
    <actuator>
    </actuator>
</mujoco>
"""

model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
viewer = MjViewer(sim)
viewer.cam.elevation = -7
viewer.cam.azimuth = 180
viewer._run_speed = 1
viewer.cam.lookat[2] = -1
sim
t = 0
while True:
    #sim.data.ctrl[0] = math.cos(t / 1000.) # * 0.01
    #sim.data.ctrl[1] = math.sin(t / 1000.) # * 0.01
    t += 1
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break