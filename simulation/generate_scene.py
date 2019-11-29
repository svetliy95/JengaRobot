from numpy.random import normal
from math import sqrt
from constants import *
from pusher import Pusher
from tower import Tower


def generate_pusher_actuator_forces_sensor():
    s = """<actuatorfrc actuator="x_pusher_actuator"/>
                <actuatorfrc actuator="y_pusher_actuator"/>
                <actuatorfrc actuator="z_pusher_actuator"/>"""
    return s


def generate_block_rotation_sensors(num):
    s = ""
    for i in range(num):
        s += f"""<framequat name="block{i}_rotation" objtype="body" objname="block{i}"/>\n                """
    return s

def generate_block_position_sensors(num):
    s = ""
    for i in range(num):
        s += f"""<framepos name="block{i}_pos" objtype="body" objname="block{i}"/>\n                """
    return s


def generate_coordinate_axes():
    s = f"""<body name="coordinate_axes" pos="{coordinate_axes_pos_x} {coordinate_axes_pos_y} {coordinate_axes_pos_z}">
                    <geom name="origin" rgba="0.33 0.33 0.33 1" pos="0 0 {coordinate_axes_width}" size="{coordinate_axes_width} {coordinate_axes_width} {coordinate_axes_width}" type="box"/>
                    <geom name="x_axis" rgba="1 0 0 1" pos="{coordinate_axes_height + coordinate_axes_width} 0 {coordinate_axes_width}" type="box" size="{coordinate_axes_height} {coordinate_axes_width} {coordinate_axes_width}"/>
                    <geom name="y_axis" rgba="0 1 0 1" pos="0 {coordinate_axes_height + coordinate_axes_width} {coordinate_axes_width}" type="box" size="{coordinate_axes_width} {coordinate_axes_height} {coordinate_axes_width}"/>
                    <geom name="z_axis" rgba="0 0 1 1" pos="0 0 {coordinate_axes_height + 2 * coordinate_axes_width}" type="box" size="{coordinate_axes_width} {coordinate_axes_width} {coordinate_axes_height}"/>
                </body>"""

    return s


def generate_block(number, pos_sigma, angle_sigma, spacing):
    # TODO spacing automatic calculation

    if number % 6 < 3:  # even level
        x = 0
        y = -block_width_mean + (number % 3) * block_width_mean
        y += (number % 3) * spacing  # add spacing between blocks
        z = number//3 * block_height_mean
        angle_z = normal(0, angle_sigma)  # add disturbance to the angle
    else:  # odd level
        x = -block_width_mean + (number % 3) * block_width_mean
        x += (number % 3) * spacing  # add spacing between blocks
        y = 0
        z = number//3 * block_height_mean
        angle_z = normal(90, angle_sigma)  # rotate and add disturbance

    # add disturbance to mass, position and sizes
    mass = normal(block_mass_mean, block_mass_sigma)
    [x, y] = normal([x, y], [pos_sigma, pos_sigma])
    [block_size_x, block_size_y, block_size_z] = normal([block_length_mean/2, block_width_mean/2, block_height_mean/2],
                                                        [block_length_sigma, block_width_sigma, block_height_sigma])

    s = f'''
                <body name="block{number}" pos="{x} {y} {z+block_height_mean/2}" euler="0 0 {angle_z}">
                    <joint type="slide" axis="1 0 0" pos ="0 0 0"/>
                    <joint type="slide" axis="0 1 0" pos ="0 0 0"/>
                    <joint type="slide" axis="0 0 1" pos ="0 0 0"/>
                    <joint type="hinge" axis="1 0 0"  pos ="0 0 0"/>
                    <joint type="hinge" axis="0 1 0" pos ="0 0 0"/>
                    <joint type="hinge" axis="0 0 1"  pos ="0 0 0"/>
                    <geom mass="{mass}" pos="0 0 0" class="block" size="{block_size_x} {block_size_y} {block_size_z}" type="box"/>
                </body>'''
    return s


def generate_scene(num_blocks=54,
                   timestep = 0.002,
                   pos_sigma=0.0005,
                   angle_sigma=0.2,
                   spacing=block_width_sigma*3):


    blocks_xml = ""
    for i in range(num_blocks):
        blocks_xml += Tower.generate_block(i, pos_sigma, angle_sigma, spacing)

    block_position_sensors_xml = generate_block_position_sensors(num_blocks)
    block_rotation_sensors_xml = generate_block_rotation_sensors(num_blocks)
    pusher_actuator_sensors_xml = generate_pusher_actuator_forces_sensor()
    MODEL_XML = f"""
        <?xml version="1.0" ?>
        <mujoco>
            
            <size nconmax="3200" njmax="8000"/>    
            <statistic extent="2" meansize=".05"/>
              
            <default class="main">
                <default class="block">
                    <geom rgba="0.8235 0.651 0.4745 1" condim="6" friction="0.4 0.005 0.0001"/>
                </default>
            </default>
            <option timestep="{timestep}" integrator="Euler" cone="elliptic" solver="Newton" o_solimp="0.999 0.999 0.01 0.5 2" o_solref="0 5" noslip_iterations="0"/>
        
            <visual>
                <rgba haze="0.15 0.25 0.35 1"/>
                <quality shadowsize="2048"/>
                <map stiffness="700" shadowscale="0.5" fogstart="10" fogend="15" zfar="40" haze="0.3"/>
            </visual>
        
            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/> 
                <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" 
                    width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
                <texture name="texsponge" type="2d" file="/home/bch_svt/cartpole/simulation/images/sponge.png"/>   
                <texture name="texcarpet" type="2d" file="/home/bch_svt/cartpole/simulation/images/carpet.png"/>  
        
                <texture name="texmarble" type="cube" file="/home/bch_svt/cartpole/simulation/images/marble.png"/>
        
                <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
                <material name="matcarpet" texture="texcarpet"/>
                <material name="matsponge" texture="texsponge" specular="0.3"/>
                <material name="matmarble" texture="texmarble" rgba=".7 .7 .7 1"/>
            </asset>
        
            <worldbody>
                <!-- lighting -->
                <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1" castshadow="false"/>
                <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>
                
                <!-- floor -->
                <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
                
                <!-- coordinate axes -->
                {generate_coordinate_axes()}
                
                <!-- pusher -->
                <body name="pusher_base" pos="{Pusher.START_X} {Pusher.START_Y} {Pusher.START_Z}">
                    <joint name="x_pusher_slide" type="slide" axis="1 0 0" damping="{Pusher.pusher_base_damping}" pos ="0 0 0"/>
                    <joint name="y_pusher_slide" type="slide" axis="0 1 0" damping="{Pusher.pusher_base_damping}" pos ="0 0 0"/>
                    <joint name="z_pusher_slide" type="slide" axis="0 0 1" damping="{Pusher.pusher_base_damping}" pos ="0 0 0"/>
                    <joint name="x_pusher_hinge" type="hinge" axis="1 0 0" damping="{Pusher.pusher_base_damping}" pos ="0 0 0"/>
                    <joint name="y_pusher_hinge" type="hinge" axis="0 1 0" damping="{Pusher.pusher_base_damping}" pos ="0 0 0"/>
                    <joint name="z_pusher_hinge" type="hinge" axis="0 0 1" damping="{Pusher.pusher_base_damping}" pos ="0 0 0"/>
                    <geom name="pusher" type="box" size="{Pusher.pusher_size*5} {Pusher.pusher_size} {Pusher.pusher_size}" mass="{Pusher.pusher_base_mass}"/>
                    <site type="box" pos="{-pusher_spring_length} 0 0" name="pusher_site"/>
                    <body name="pusher" pos="{-pusher_spring_length} 0 0">
                        <joint name="x_dummy_slide" type="slide" axis="1 0 0" stiffness="{Pusher.pusher_kp}" damping="{Pusher.pusher_damping}" pos ="0 0 0"/>               
                        <geom type="box" size="{Pusher.pusher_size} {Pusher.pusher_size} {Pusher.pusher_size}" mass="{Pusher.pusher_mass}"/>                        
                    </body>
                </body>
                
                <!-- jenga blocks -->
                {blocks_xml} 
        
            </worldbody>
            
            <actuator>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="x_pusher_slide" name="x_pusher_actuator"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="y_pusher_slide" name="y_pusher_actuator"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="z_pusher_slide" name="z_pusher_actuator"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="x_pusher_hinge"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="y_pusher_hinge"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="z_pusher_hinge"/>
            </actuator>
            
            <sensor>
                
                <!-- block position sensors -->
                {block_position_sensors_xml}
                <!-- block rotation sensors -->
                {block_rotation_sensors_xml}
                <force site="pusher_site"/>
                <!-- pusher force sensors -->
                <!-- {pusher_actuator_sensors_xml} -->
            </sensor>
        </mujoco>
        """

    return MODEL_XML


if __name__ == "__main__":
    string = generate_scene()
    print(string)
    file = open("/home/bch_svt/.mujoco/mujoco200/model/jenga.xml", "w")
    file.write(string)