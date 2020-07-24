from mujoco_py import load_model_from_xml, MjSim, MjViewer
import os
import numpy as np
from threading import Thread
import time
from cv.blocks_calibration import *
import matplotlib; matplotlib.use("TkAgg")
import math
from cv.transformations import matrix2pose_XYZ, pose2matrix_XYZ, get_Rz_h, get_Ry_h, get_Rx_h


class CoordinateSystem:
    def __init__(self, vx, vy, vz, origin):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.origin = origin

    # for ratation
    def get_transformation_mtx(self):
        B = np.array([self.vx, self.vy, self.vz, self.origin]).transpose()
        B = np.concatenate((B, np.reshape([0, 0, 0, 1], (1, 4))), axis=0)
        A = np.eye(4)
        T = np.linalg.inv(A) @ B

        return T

def generate_scene(scaler, coord_ax_pos, coord_ax_euler):
    print(scaler)
    # coordinate axes
    coordinate_axes_width = 0.05
    coordinate_axes_height = 0.3

    168, 50, 123
    red = np.array([168, 50, 123]) / 255
    green = np.array([123, 168, 50]) / 255
    blue = np.array([50, 142, 168]) / 255

    gripper = f"""<body name="gripper" pos="0 0 0" euler="0 0 0" mocap="true">
                        <geom rgba="0.33 0.33 0.33 1" pos="0 0 {coordinate_axes_width}" size="{coordinate_axes_width} {coordinate_axes_width} {coordinate_axes_width}" type="box"/>
                        <geom rgba="{red[0]} {red[1]} {red[2]} 1" pos="{coordinate_axes_height + coordinate_axes_width} 0 {coordinate_axes_width}" type="box" size="{coordinate_axes_height} {coordinate_axes_width} {coordinate_axes_width}"/>
                        <geom rgba="{green[0]} {green[1]} {green[2]} 1" pos="0 {coordinate_axes_height + coordinate_axes_width} {coordinate_axes_width}" type="box" size="{coordinate_axes_width} {coordinate_axes_height} {coordinate_axes_width}"/>
                        <geom rgba="{blue[0]} {blue[1]} {blue[2]} 1" pos="0 0 {coordinate_axes_height + 2 * coordinate_axes_width}" type="box" size="{coordinate_axes_width} {coordinate_axes_width} {coordinate_axes_height}"/>
                    </body>"""

    red = np.array([235, 64, 52]) / 255
    green = np.array([50, 168, 82]) / 255
    blue = np.array([66, 135, 245]) / 255

    coord_ax_custom = f"""<body name="coordinate_axes_custom" pos="{coord_ax_pos[0]} {coord_ax_pos[1]} {coord_ax_pos[2]}" euler="{coord_ax_euler[0]} {coord_ax_euler[1]} {coord_ax_euler[2]}">
                    <geom name="origin" rgba="0.33 0.33 0.33 1" pos="0 0 {coordinate_axes_width}" size="{coordinate_axes_width} {coordinate_axes_width} {coordinate_axes_width}" type="box"/>
                    <geom name="x_axis" rgba="{red[0]} {red[1]} {red[2]} 1" pos="{coordinate_axes_height + coordinate_axes_width} 0 {coordinate_axes_width}" type="box" size="{coordinate_axes_height} {coordinate_axes_width} {coordinate_axes_width}"/>
                    <geom name="y_axis" rgba="{green[0]} {green[1]} {green[2]} 1" pos="0 {coordinate_axes_height + coordinate_axes_width} {coordinate_axes_width}" type="box" size="{coordinate_axes_width} {coordinate_axes_height} {coordinate_axes_width}"/>
                    <geom name="z_axis" rgba="{blue[0]} {blue[1]} {blue[2]} 1" pos="0 0 {coordinate_axes_height + 2 * coordinate_axes_width}" type="box" size="{coordinate_axes_width} {coordinate_axes_width} {coordinate_axes_height}"/>
                </body>"""



    coord_ax_global = f"""<body name="coordinate_axes_global" pos="0 0 0" euler="0 0 0">
                        <geom  rgba="0.33 0.33 0.33 1" pos="0 0 {coordinate_axes_width}" size="{coordinate_axes_width} {coordinate_axes_width} {coordinate_axes_width}" type="box"/>
                        <geom  rgba="1 0 0 1" pos="{coordinate_axes_height + coordinate_axes_width} 0 {coordinate_axes_width}" type="box" size="{coordinate_axes_height} {coordinate_axes_width} {coordinate_axes_width}"/>
                        <geom  rgba="0 1 0 1" pos="0 {coordinate_axes_height + coordinate_axes_width} {coordinate_axes_width}" type="box" size="{coordinate_axes_width} {coordinate_axes_height} {coordinate_axes_width}"/>
                        <geom  rgba="0 0 1 1" pos="0 0 {coordinate_axes_height + 2 * coordinate_axes_width}" type="box" size="{coordinate_axes_width} {coordinate_axes_width} {coordinate_axes_height}"/>
                    </body>"""



    res = f"""
    <?xml version="1.0" ?>
    <mujoco>
        <size nconmax="3200" njmax="8000"/>    
        <statistic extent="2" meansize=".05"/>

        <default class="main">
            <default class="block">
                <geom rgba="0.8235 0.651 0.4745 1" condim="6" friction="0.4 0.005 0.0001"/>
            </default>
        </default>
        <option timestep="0.005" />

        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <quality shadowsize="2048"/>
            <map stiffness="700" shadowscale="0.5" fogstart="10" fogend="15" zfar="40" haze="0.3"/>
        </visual>

        <asset>
            <texture name="tex_floor_tag" type="cube" filefront="/home/bch_svt/cartpole/simulation/cv/textures/floor_tag.png"/>
            <material name="mat_floor_tag" texture="tex_floor_tag"/>
            <texture name="tex_shield_front" type="cube" fileright="/home/bch_svt/cartpole/simulation/cv/textures/shield_right.png"/>
            <material name="mat_shield_front" texture="tex_shield_front"/>
            <texture name="tex_shield_back" type="cube" fileleft="/home/bch_svt/cartpole/simulation/cv/textures/shield_left.png"/>
            <material name="mat_shield_back" texture="tex_shield_back"/>
            <texture name="tex_block_top" type="cube" fileup="/home/bch_svt/cartpole/simulation/cv/textures/top.png"/>
            <material name="mat_block_top" texture="tex_block_top"/>
            <texture name="tex_cali" type="cube" fileright="/home/bch_svt/cartpole/simulation/images/cali_tag.png"/>
            <material name="mat_cali" texture="tex_cali"/>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/> 
            <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" 
                width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
        </asset>

        <worldbody>
            <!-- lighting -->
            <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="2 0 5.0" dir="0 0 -1" castshadow="false"/>
            <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="4 0 4" dir="-1 0 -1"/>
            <!-- floor -->
            <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>

            {coord_ax_global}
            {coord_ax_custom}

            {gripper}

        </worldbody>
    </mujoco>
    """

    return res





if __name__ == '__main__':
    scaler = 0.01

    # coord system
    coord_system_pos = np.array([100, 100, 0]) * scaler
    coord_system_euler = np.array([-13, 42, 75])
    cs_quat = Quaternion(axis=x_unit_vector, degrees=coord_system_euler[0]) * \
                   Quaternion(axis=y_unit_vector, degrees=coord_system_euler[1]) * \
                   Quaternion(axis=z_unit_vector, degrees=coord_system_euler[2])
    cs_pose = np.concatenate((coord_system_pos, np.array(list(map(math.degrees,cs_quat.yaw_pitch_roll[::-1])))))
    cs_pose = np.concatenate((coord_system_pos, np.array(cs_quat.yaw_pitch_roll[::-1])))
    print(f"CS pose: {cs_pose}")
    print(f"CS eulers: {list(map(math.degrees,cs_quat.yaw_pitch_roll[::-1]))}")
    print(f"CS mtx from pose:")
    print(f"{pose2matrix_XYZ(cs_pose)}")

    # initialize gripper
    gripper_pos = np.array([200, 200, 10]) * scaler
    gripper_euler_degrees = np.array([0, 0, 0])
    gripper_quat = Quaternion(axis=x_unit_vector, degrees=gripper_euler_degrees[0]) * \
                   Quaternion(axis=y_unit_vector, degrees=gripper_euler_degrees[1]) * \
                   Quaternion(axis=z_unit_vector, degrees=gripper_euler_degrees[2])
    gripper_euler_radians = gripper_euler_degrees / 180 * math.pi
    gripper_pose = np.concatenate((gripper_pos, gripper_euler_radians))
    gripper_mtx = pose2matrix_XYZ(gripper_pose)

    # get transformation matrix
    cs = CoordinateSystem(cs_quat.rotate(x_unit_vector), cs_quat.rotate(y_unit_vector), cs_quat.rotate(z_unit_vector), coord_system_pos)
    print(f"Transformation mtx:\n {cs.get_transformation_mtx()}")
    print(f"Transformation mtx pose: {matrix2pose_XYZ(cs.get_transformation_mtx())}")
    print(f"Gripper mtx:\n {gripper_mtx}")

    # transform coordinates
    gripper_mtx = cs.get_transformation_mtx() @ gripper_mtx

    # flip gripper
    gripper_mtx = gripper_mtx @ get_Rz_h(180, 'degrees')

    print(f"Gripper mtx #2:\n {gripper_mtx}")
    gripper_pose = matrix2pose_XYZ(gripper_mtx)
    print(f"Before transformation: {gripper_pose}")
    gripper_pose = matrix2pose_XYZ(gripper_mtx)
    print(f"After transformation: {gripper_pose}")
    print(f"Gripper pose: {gripper_pose[3:] / math.pi * 180}")
    gripper_pos = gripper_pose[:3]
    gripper_euler_radians = gripper_pose[3:]
    gripper_euler_degrees = gripper_euler_radians / math.pi * 180
    gripper_quat = Quaternion(axis=z_unit_vector, degrees=gripper_euler_degrees[2]) * \
                   Quaternion(axis=y_unit_vector, degrees=gripper_euler_degrees[1]) * \
                   Quaternion(axis=x_unit_vector, degrees=gripper_euler_degrees[0])





    # gripper_quat = Quaternion(matrix=gripper_mtx)



    cs_pose = np.array([3, 4, 5, 1.6, 0.3, -1.14])
    cs_mtx = pose2matrix_XYZ(cs_pose)
    cs_quat = Quaternion(matrix=cs_mtx)
    cs = CoordinateSystem(cs_quat.rotate(x_unit_vector), cs_quat.rotate(y_unit_vector), cs_quat.rotate(z_unit_vector), cs_pose[:3])

    # initialize simulation related variables
    MODEL_XML = generate_scene(scaler, coord_system_pos, coord_system_euler)
    model = load_model_from_xml(MODEL_XML)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 180
    viewer.cam.lookat[0:3] = [-1, 0, 0]
    viewer._run_speed = 16
    viewer.cam.distance = 3
    t = 0

    # simulate
    while True:
        t += 1

        if t % 100 == 0:
            sim.data.set_mocap_pos(f'gripper', gripper_pos)
            sim.data.set_mocap_quat(f'gripper', gripper_quat.q)

        sim.step()

        # update marker pos
        if t % 100 == 0:
            # estimated_shield_pos = calculate_errors(sim, take_screenshot(), 107)
            pass
        # viewer.add_marker(pos=estimated_shield_pos,
        #                   label=str(t))

        # take screenshots and calculate errors

        # render frame
        viewer.render()

        if t > 100 and os.getenv('TESTING') is not None:
            break
