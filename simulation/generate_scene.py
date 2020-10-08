from numpy.random import normal
from math import sqrt
from constants import *
from pusher import Pusher
from tower import Tower
from extractor import Extractor
import math
import scipy.stats as stats
import time
import os


def generate_textures_and_materials_assets():
    s = ''
    for i in range(g_blocks_num):
        s += f"""<texture name="text_block{i}" type="cube" fileright="/home/bch_svt/cartpole/simulation/images/texture_block{i}_right.png" fileleft="/home/bch_svt/cartpole/simulation/images/texture_block{i}_left.png"/>
                <material name="mat_block{i}" texture="text_block{i}"/>
                """
    return s


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


def generate_block(number, pos_sigma, angle_sigma, spacing, seed):
    # print(f"Seed: {seed}, pid: {os.getpid()}")
    # TODO spacing automatic calculation

    # fancy method to selecting the next seed
    random_generator = np.random.Generator(np.random.PCG64(seed**23 % 10**9))


    a = (block_height_min - block_height_mean) / block_height_sigma
    b = (block_height_max - block_height_mean) / block_height_sigma
    height_distribution = stats.truncnorm(a, b, loc=block_height_mean, scale=block_height_sigma)
    height_distribution.random_state = np.random.RandomState(seed=seed)
    # jump to the next seed
    seed = seed**29 % 10**9


    a = (block_width_min - block_width_mean) / block_width_sigma
    b = (block_width_max - block_width_mean) / block_width_sigma
    width_distribution = stats.truncnorm(a, b, loc=block_width_mean, scale=block_width_sigma)
    width_distribution.random_state = np.random.RandomState(seed=seed)
    # jump to next seed
    seed = seed ** 29 % 10 ** 9

    a = (block_length_min - block_length_mean) / block_length_sigma
    b = (block_length_max - block_length_mean) / block_length_sigma
    length_distribution = stats.truncnorm(a, b, loc=block_length_mean, scale=block_length_sigma)
    length_distribution.random_state = np.random.RandomState(seed=seed)
    # jump to next seed
    seed = seed ** 29 % 10 ** 9

    if number % 6 < 3:  # even level
        x = 0
        y = -block_width_mean + (number % 3) * block_width_mean
        y += (number % 3) * spacing  # add spacing between blocks
        z = number // 3 * block_height_mean + block_height_mean / 2
        angle_z = random_generator.normal(0, angle_sigma)  # add disturbance to the angle
    else:  # odd level
        x = -block_width_mean + (number % 3) * block_width_mean
        x += (number % 3) * spacing  # add spacing between blocks
        y = 0
        z = number // 3 * block_height_mean + block_height_mean / 2
        angle_z = random_generator.normal(90, angle_sigma)  # rotate and add disturbance

    # add disturbance to mass, position and sizes
    mass = random_generator.normal(block_mass_mean, block_mass_sigma)
    x = random_generator.normal(x, pos_sigma)
    y = random_generator.normal(y, pos_sigma)
    [block_size_x, block_size_y, block_size_z] = [length_distribution.rvs()[0] / 2, width_distribution.rvs()[0] / 2,
                                                  height_distribution.rvs()[0] / 2]


    # WARNING: debugging code!
    if number == 0:
        print("The size of the first block is changed!")
        block_size_z = (block_height_min / 2) * 0.99
        mass = block_mass_mean * 100

    Tower.block_sizes.append(np.array([block_size_x * 2, block_size_y * 2, block_size_z * 2]))
    s = f'''
                <body name="block{number}" pos="{x} {y} {z}" euler="0 0 {angle_z}">
                    <freejoint name="{Tower.block_prefix + "_" + str(number) + "_joint"}"/>
                    <geom mass="{mass}" pos="0 0 0" class="block" size="{block_size_x} {block_size_y} {block_size_z}" type="box" material="mat_block{number}"/>
                </body>'''
    return s


def generate_scene(num_blocks=54,
                   timestep = 0.002,
                   pos_sigma=0.0005,
                   angle_sigma=0.2,
                   spacing=block_width_sigma*3,
                   seed=None):

    print(f"Process seed: {seed}")

    if seed is None:
        initial_seed = time.time_ns()
        seed = initial_seed
    else:
        initial_seed = seed

    print(f"PID: {os.getpid()}: {seed}")
    blocks_xml = ""
    for i in range(num_blocks):
        # Choose the next seed. 19 is a prime number and
        seed = seed ** 19 % 10 ** 9
        blocks_xml += generate_block(i, pos_sigma, angle_sigma, spacing, seed)



    block_position_sensors_xml = generate_block_position_sensors(num_blocks)
    block_rotation_sensors_xml = generate_block_rotation_sensors(num_blocks)
    pusher_actuator_sensors_xml = generate_pusher_actuator_forces_sensor()
    textures_and_material_assets = generate_textures_and_materials_assets()
    MODEL_XML = f"""
        <?xml version="1.0" ?>
        <mujoco>
            
            <!-- <size nconmax="3200" njmax="8000"/> -->    
            <size nconmax="1600" njmax="4000"/>    
            <statistic extent="2" meansize=".05"/>
              
            <default class="main">
                <default class="block">
                    <geom rgba="0.8235 0.651 0.4745 1" condim="6" friction="0.4 0.005 0.0001"/>
                </default>
            </default>
            
            <!-- customized default -->
            <option timestep="{timestep}" integrator="Euler" cone="elliptic" solver="Newton" impratio="10" o_solimp="0.9 0.95 0.001 0.5 2" o_solref="0 2" noslip_iterations="0"/> 
            <!-- -->
            <!--
            <option timestep="{timestep}" integrator="Euler" cone="elliptic" solver="Newton" o_solimp="0.999 0.999 0.01 0.5 2" o_solref="0 5" noslip_iterations="50"/>
            -->
        
            <visual>
                <rgba haze="0.15 0.25 0.35 1"/>
                <quality shadowsize="2048"/>
                <map stiffness="700" shadowscale="0.5" fogstart="10" fogend="15" zfar="40" haze="0.3"/>
            </visual>
        
            <asset>
                <!-- textures and corresponding materials for blocks -->
                {textures_and_material_assets}
                <texture name="tex_floating" type="cube" fileright="/home/bch_svt/cartpole/simulation/images/floating_object_texture.png"/>
                <material name="mat_floating" texture="tex_floating"/>
                <texture name="tex_coord_frame" type="cube" filefront="/home/bch_svt/cartpole/simulation/images/coordinate_frame_texture.png"/>
                <material name="mat_coord_frame" texture="tex_coord_frame"/>
                <texture name="tex_coord_frame2" type="cube" filefront="/home/bch_svt/cartpole/simulation/images/coordinate_frame_texture2.png"/>
                <material name="mat_coord_frame2" texture="tex_coord_frame2"/>
                <texture name="tex_coord_frame3" type="cube" filefront="/home/bch_svt/cartpole/simulation/images/coordinate_frame_texture3.png"/>
                <material name="mat_coord_frame3" texture="tex_coord_frame3"/>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/> 
                <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" 
                    width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
                <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
            </asset>
        
            <worldbody>
                <!-- cameras -->
                <camera name="cam1" mode="fixed" pos="13.619144207633779 -12.881811375734092 10.253618753368848" xyaxes="13.697094885436288 13.303969412909323 -0.0 -50.36766872056147 51.85600749760638 364.60601044027175" fovy="45"/>
                <camera name="cam2" mode="fixed" pos="-12.771330489383502 13.023560783310552 10.379449117055714" xyaxes="-12.211118677158598 -13.083650787812111 0.0 47.17238071883744 -44.02651435625462 320.293337285068" fovy="45"/>
                
                <!-- lighting -->
                <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1" castshadow="false"/>
                <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>
                <!-- floor -->
                <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
                
                <!-- coordinate system tag -->
                <geom name="coordinate_frame_tag" type="box" size="{coordinate_frame_tag_size[0]/2} {coordinate_frame_tag_size[1]/2} {coordinate_frame_tag_size[2]/2}" pos="{coordinate_frame_tag_pos[0]} {coordinate_frame_tag_pos[1]} {coordinate_frame_tag_pos[2]}" rgba="1 1 1 1" material="mat_coord_frame" euler="0 0 90"/>                   
                
                <!-- coordinate axes -->
                {generate_coordinate_axes()}
                `
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
                
                
                <!-- floating body with tag -->
                <!--
                <body name="floating_body" pos="-7 0 3" mocap="true">
                    <geom name="floating" type="box" size="{0.05*scaler} {0.05*scaler} {0.05*scaler}" mass="{Pusher.pusher_base_mass}" material="mat_floating" euler= "0 0 0"/>
                </body>
                
                -->
                
                <!-- extractor -->
                {Extractor.generate_xml()}
                
                <!-- zwischenablage -->
                <body name="zwischenablage" 
                      pos="{zwischanablage_pos[0]} {zwischanablage_pos[1]} {zwischanablage_pos[2]}" 
                      quat="{zwischenablage_quat_elem[0]} {zwischenablage_quat_elem[1]} {zwischenablage_quat_elem[2]} {zwischenablage_quat_elem[3]}">
                    <geom name="base" type="box" friction="0 0 0" pos="0 0 0" size="{zwischanablage_base_size[0]} {zwischanablage_base_size[1]} {zwischanablage_base_size[2]}"/>
                    <geom name="bottom_wall"
                          friction="0 0 0" 
                          type="box"
                          pos="{zwischanablage_base_size[0] + zwischanablage_bottom_wall_size[0]} 0 {-zwischanablage_base_size[2] + zwischanablage_bottom_wall_size[2]}" 
                          size="{zwischanablage_bottom_wall_size[0]} {zwischanablage_bottom_wall_size[1]} {zwischanablage_bottom_wall_size[2]}"/>
                    <geom name="side_wall"
                          friction="0 0 0"
                          type="box" 
                          pos="0 {-zwischanablage_base_size[1] - zwischanablage_side_wall_size[1]} {-zwischanablage_base_size[2] + zwischanablage_side_wall_size[2]}" 
                          size="{zwischanablage_side_wall_size[0]} {zwischanablage_side_wall_size[1]} {zwischanablage_side_wall_size[2]}"/>
                </body>             
        
            </worldbody>
            
            <actuator>
                <!-- pusher actuators -->
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="x_pusher_slide" name="x_pusher_actuator"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="y_pusher_slide" name="y_pusher_actuator"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="z_pusher_slide" name="z_pusher_actuator"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="x_pusher_hinge"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="y_pusher_hinge"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="z_pusher_hinge"/>
                
                <!-- extractor actuators -->
                <!-- position -->
                <position name="extractor_slide_x_actuator" kp="{Extractor.base_kp}" gear="1 0 0 0 0 0" joint="extractor_slide_x"/>
                <position name="extractor_slide_y_actuator" kp="{Extractor.base_kp}" gear="1 0 0 0 0 0" joint="extractor_slide_y"/>
                <position name="extractor_slide_z_actuator" kp="{Extractor.base_kp}" gear="1 0 0 0 0 0" joint="extractor_slide_z"/>
                <position name="extractor_hinge_x_actuator" kp="{Extractor.base_kp}" gear="1 0 0 0 0 0" joint="extractor_hinge_x"/>
                <position name="extractor_hinge_y_actuator" kp="{Extractor.base_kp}" gear="1 0 0 0 0 0" joint="extractor_hinge_y"/>
                <position name="extractor_hinge_z_actuator" kp="{Extractor.base_kp}" gear="1 0 0 0 0 0" joint="extractor_hinge_z"/>
                <!-- fingers -->
                <position name="finger1_actuator" kp="{Extractor.finger_kp}" gear="1 0 0 0 0 0" joint="finger1_joint"/>
                <position name="finger2_actuator" kp="{Extractor.finger_kp}" gear="1 0 0 0 0 0" joint="finger2_joint"/>
                
                <!--
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="x_floating_slide"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="y_floating_slide"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="z_floating_slide"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="x_floating_hinge"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="y_floating_hinge"/>
                <position kp="{Pusher.pusher_base_kp}" gear="1 0 0 0 0 0" joint="z_floating_hinge"/>
                -->
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

    return MODEL_XML, initial_seed


if __name__ == "__main__":
    string = generate_scene(timestep=g_timestep)
    print(string)
    file = open("/home/bch_svt/.mujoco/mujoco200/model/jenga.xml", "w")
    file.write(string)