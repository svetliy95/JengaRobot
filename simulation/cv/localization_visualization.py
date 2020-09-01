from mujoco_py import load_model_from_xml, MjSim, MjViewer
import os
import numpy as np
from cv.camera import Camera
from constants import *
from cv.block_localization import get_block_positions
from cv.transformations import matrix2pose_ZYX
from threading import Thread
import time
from cv.blocks_calibration import *
from copy import deepcopy
import dt_apriltags
import math
from cv.blocks_calibration import read_corrections, correct_poses
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib; matplotlib.use("TkAgg")

def generate_block_xml(id, size, cali: bool):
    length = size['length'] * scaler
    width = size['width'] * scaler
    height = size['height'] * scaler
    shield_height = 12 * scaler
    shield_width = 22 * scaler
    shield_thickness = 1 * scaler
    shield_x_size = shield_thickness / 2
    shield_y_size = shield_width / 2
    shield_z_size = shield_height / 2
    block_x_size = (length - 2 * shield_thickness) / 2
    block_y_size = width / 2
    block_z_size = height / 2

    if cali:
        res = f"""<body name="block_{str(id)}" pos="{0} {0} {5 + id}" euler="0 0 0" mocap="true">
                <geom name="block_{str(id)}" pos="0 0 0" size="{block_x_size} {block_y_size} {block_z_size}" type="box" rgba="0.8235 0.651 0.4745 1"/>
                <geom name="shield_left_{str(id)}" pos="{-(block_x_size + shield_thickness/2)} 0 0" size="{shield_x_size} {shield_y_size} {shield_z_size}" type="box"/>
                <geom name="shield_right_{str(id)}" pos="{block_x_size + shield_thickness/2} 0 0" size="{shield_x_size} {shield_y_size} {shield_z_size}" type="box"/>
            </body>\n"""
    else:
        if id < 54:
            res = f"""<body name="block_{str(id)}" pos="{0} {0} {5 + id}" euler="0 0 0" mocap="true">
                            <geom name="block_{str(id)}" pos="0 0 0" size="{block_z_size} {block_y_size} {block_x_size}" type="box" rgba="0.8235 0.651 0.4745 1"/>
                            <geom name="shield_left_{str(id)}" pos="0 0 {-(block_x_size + shield_thickness / 2)}" size="{shield_z_size} {shield_y_size} {shield_x_size}" type="box" rgba="0 0 1 1" material="mat_shield_back"/>
                            <geom name="shield_right_{str(id)}" pos="0 0 {block_x_size + shield_thickness / 2}" size="{shield_z_size} {shield_y_size}  {shield_x_size}" type="box" rgba="0 1 0.4745 1" material="mat_shield_front"/>
                        </body>\n"""
            res = f"""<body name="block_{str(id)}" pos="{0} {0} {5 + id}" euler="0 0 0" mocap="true">
                            <geom name="block_{str(id)}" pos="0 0 0" size="{block_x_size} {block_y_size} {block_z_size}" type="box" rgba="0.8235 0.651 0.4745 1" material="mat_block_top"/>
                            <geom name="shield_left_{str(id)}" pos="{-(block_x_size + shield_thickness / 2)} 0 0" size="{shield_x_size} {shield_y_size} {shield_z_size}" type="box" material="mat_shield_back"/>
                            <geom name="shield_right_{str(id)}" pos="{block_x_size + shield_thickness / 2} 0 0" size="{shield_x_size} {shield_y_size} {shield_z_size}" type="box" material="mat_shield_front"/>
                        </body>\n"""
        else:
            res = f"""<body name="block_{str(id)}" pos="{0} {0} {5 + id}" euler="0 0 0" mocap="true">
                                        <geom name="block_{str(id)}" pos="0 0 0" size="{block_z_size} {block_y_size} {block_x_size}" type="box" rgba="0.75 0.651 0.6 1"/>
                                        <geom name="shield_left_{str(id)}" pos="0 0 {-(block_x_size + shield_thickness / 2)}" size="{shield_z_size} {shield_y_size} {shield_x_size}" type="box" rgba="0 0 1 1" material="mat_shield_back"/>
                                        <geom name="shield_right_{str(id)}" pos="0 0 {block_x_size + shield_thickness / 2}" size="{shield_z_size} {shield_y_size}  {shield_x_size}" type="box" rgba="0 1 0.4745 1" material="mat_shield_front"/>
                                    </body>\n"""

    return res

def generate_shield_materials():
    res = ''
    for i in range(54):
        res += f"""\t\t<texture name="texture_block{i}_left_shield" type="cube" fileright="/home/bch_svt/cartpole/simulation/cv/textures/texture_block{i}_left_shield.png"/>
            <material name="mat_block{i}_left_shield" texture="texture_block{i}_left_shield"/>
            <texture name="texture_block{i}_right_shield" type="cube" fileleft="/home/bch_svt/cartpole/simulation/cv/textures/texture_block{i}_right_shield.png"/>
            <material name="mat_block{i}_right_shield" texture="texture_block{i}_right_shield"/>\n"""
    print(res)
    return res

def set_block_color(id, color, sim):
    geom_id = sim.model.geom_name2id(f'block_{id}')
    sim.model.geom_rgba[geom_id] = color

def set_shield_color(id, side, color, sim):
    geom_id = sim.model.geom_name2id(f'shield_{side}_{id}')
    sim.model.geom_rgba[geom_id] = color

def generate_blocks_xml(sizes, cali: bool):
    res = ''
    for i in sizes:
        block_xml = generate_block_xml(i, sizes[i], cali)
        res += block_xml

    return res

def generate_scene_with_cali(scaler, cali_pos, shield_pose, cali_fl: bool):

    # coordinate axes
    coordinate_axes_pos_x = -2
    coordinate_axes_pos_y = -2
    coordinate_axes_pos_z = 0
    coordinate_axes_width = 0.05
    coordinate_axes_height = 0.3

    shild = f"""<body name="shield" pos="{shield_pose[0]} {shield_pose[1]} {shield_pose[2]}" euler="0 0 0" mocap="true">
                    <geom name="shield" pos="{-1*scaler} 0 0" size="{1*scaler} {22*scaler} {12*scaler}" type="box" material="mat_shield"/>
                </body>"""

    coord_ax = f"""<body name="coordinate_axes" pos="{coordinate_axes_pos_x} {coordinate_axes_pos_y} {coordinate_axes_pos_z}">
                    <geom name="origin" rgba="0.33 0.33 0.33 1" pos="0 0 {coordinate_axes_width}" size="{coordinate_axes_width} {coordinate_axes_width} {coordinate_axes_width}" type="box"/>
                    <geom name="x_axis" rgba="1 0 0 1" pos="{coordinate_axes_height + coordinate_axes_width} 0 {coordinate_axes_width}" type="box" size="{coordinate_axes_height} {coordinate_axes_width} {coordinate_axes_width}"/>
                    <geom name="y_axis" rgba="0 1 0 1" pos="0 {coordinate_axes_height + coordinate_axes_width} {coordinate_axes_width}" type="box" size="{coordinate_axes_width} {coordinate_axes_height} {coordinate_axes_width}"/>
                    <geom name="z_axis" rgba="0 0 1 1" pos="0 0 {coordinate_axes_height + 2 * coordinate_axes_width}" type="box" size="{coordinate_axes_width} {coordinate_axes_width} {coordinate_axes_height}"/>
                </body>"""

    floor_tag = f"""<body name="floor_tag" pos="{0} {0} {0}">
                    <geom name="floor_tag" pos="0 0 {1*scaler}" size="{50/8*10/2*scaler} {50/8*10/2*scaler} {1*scaler}" type="box" material="mat_floor_tag"/>
                </body>"""

    cali = ""
    if cali_fl:
        cali = f'''<body name="cali" pos="{cali_pos[0]} {cali_pos[1]} {cali_pos[2]}" euler="0 0 0">
                        <!-- tag plane -->
                        <geom type="box" pos="{-7*scaler*0.5} 0 0" size="{7*scaler*0.5} {50*scaler*0.5} {50*scaler*0.5}" mass="{1}" material="mat_cali"/>
                        <!-- bottom -->
                        <geom type="box" pos="{-85*scaler*0.5} {75*scaler*0.5} {-45*scaler*0.5}" size="{85*scaler*0.5} {35*scaler*0.5} {5*scaler*0.5}" mass="{1}"/>
                        <!-- side wall -->
                        <geom type="box" pos="{-92*scaler*0.5} {45*scaler*0.5} {-25*scaler*0.5}" size="{78*scaler*0.5} {5*scaler*0.5} {15*scaler*0.5}" mass="{1}"/>
                        <!-- back wall -->
                        <geom type="box" pos="{-165*scaler*0.5} {80*scaler*0.5} {-25*scaler*0.5}" size="{5*scaler*0.5} {30*scaler*0.5} {15*scaler*0.5}" mass="{1}"/>
                        <!-- bottom border -->
                        <geom type="box" pos="{-1*scaler*0.5} {(50+1.5+30)*scaler*0.5} {-(50-10-1.4)*scaler*0.5}" size="{1*scaler*0.5} {28.5*scaler*0.5} {1.4*scaler*0.5}" mass="{1}"/>
                        <!-- left border -->
                        <geom type="box" pos="{-1*scaler*0.5} {51.5*scaler*0.5} {-(50-10-15)*scaler*0.5}" size="{1*scaler*0.5} {1.5*scaler*0.5} {15*scaler*0.5}" mass="{1}"/>                   
                    </body>'''



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
            <texture name="tex_block_top" type="cube" fileup="/home/bch_svt/cartpole/simulation/cv/textures/top.png" filedown="/home/bch_svt/cartpole/simulation/cv/textures/top.png"/>
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
            
            {floor_tag}
            
            {coord_ax}
            
            {cali}
            
            {generate_blocks_xml(block_sizes, cali_fl)}
                       
        </worldbody>
    </mujoco>
    """

    return res

def get_shild_pose(tag_id, cam, detector, corrections):
    global shield_pos, shield_quat

    start = time.time()
    im = cam.take_picture()
    stop = time.time()
    # print(f"Execution time: {(stop-start)*1000}ms")

    cam_params = cam.get_params()
    start = time.time()
    poses = get_tag_poses_from_image(im, [106, 107], 9.6, 224, 40, np.array([0, 0, 0]), cam_params, detector)
    stop = time.time()
    # print(f"Execution time: {(stop-start)*1000}ms")

    if poses is not None and tag_id in poses:
        mtx = poses[tag_id]

        # correct pose
        pos = matrix2pose_ZYX(mtx)[:3]
        quat = Quaternion(matrix=mtx)
        measured_poses = {str(tag_id): {'pos': pos, 'quat': quat}}
        corrected_poses = correct_poses(measured_poses, corrections)
        shield_quat = corrected_poses[str(tag_id)]['quat']
        shield_pos = corrected_poses[str(tag_id)]['pos']

        shield_quat = Quaternion([shield_quat[0], -shield_quat[3], shield_quat[1], -shield_quat[2]])

        x = -shield_pos[2]*scaler*2
        y = shield_pos[0]*scaler*2
        z = -shield_pos[1]*scaler*2

        shield_pos = np.array([x, y, z])

        return shield_pos, shield_quat

def set_block_poses(cam1, cam2, detector, block_sizes):
    cam_params1 = cam1.get_params()
    cam_params2 = cam2.get_params()
    im1 = cam1.take_picture()
    im2 = cam2.take_picture()
    blank_image = np.zeros((im2.shape[0], im2.shape[1]), np.uint8)
    block_ids = [i for i in range(54)]
    poses = get_block_positions(im1, im2, block_ids, target_tag_size, ref_tag_size, ref_tag_id, ref_tag_pos, block_sizes, corrections,
                        cam_params1, cam_params2, False, detector)

    print(poses)

    if poses is not None:
        for id in range(54):
            if id in poses:
                block_pos = poses[id]['pos']
                block_quat = poses[id]['orientation']

                if calibration_mode:
                    block_pos, block_quat = swap_coordinates_cali(block_pos, block_quat)
                else:
                    block_pos, block_quat = swap_coordinates_normal(block_pos, block_quat)

                print(block_pos)

                sim.data.set_mocap_pos(f'block_{id}', block_pos)
                sim.data.set_mocap_quat(f'block_{id}', block_quat.q)

def set_block_poses_debug(cam1, cam2, detector1, detector2, block_sizes, cali_fl):
    cam_params1 = cam1.get_params()
    cam_params2 = cam2.get_params()
    start = time.time()
    im1 = cam1.get_raw_image()
    elapsed = time.time() - start
    print(f'Cam1 image time: {elapsed*1000:.2f}ms')
    start = time.time()
    im2 = cam2.get_raw_image()
    elapsed = time.time() - start
    print(f'Cam2 image time: {elapsed * 1000:.2f}ms')
    blank_image = np.zeros((im2.shape[0], im2.shape[1]), np.uint8)
    block_ids = [i for i in range(54)]
    start = time.time()
    poses1 = get_block_positions(im1, im2, block_ids, target_tag_size, ref_tag_size, ref_tag_id, ref_tag_pos, block_sizes, corrections,
                        cam_params1, cam_params2, False, detector1, detector2, cam1_mtx, cam1_dist, cam2_mtx, cam2_dist)
    elapsed = time.time() - start
    print(f'Detection time: {elapsed * 1000:.2f}ms')

    # poses2 = get_block_positions(blank_image, im2, block_ids, target_tag_size, ref_tag_size, ref_tag_id, ref_tag_pos,
    #                             block_sizes, corrections,
    #                             cam_params1, cam_params2, False, detector)

    if last_poses:
        displacements = []
        for i in poses1:
            d = np.linalg.norm(poses1[i]['pos'][:2] - last_poses[i]['pos'][:2])
            displacements.append(d)

        displacements.sort()
        max_displacements.append(max(displacements))
        print(f"Displacements:")
        print(displacements)

    for i in poses1:
        last_poses[i] = poses1[i]

    tag_height = np.array([0, 0, 2]) * scaler
    if poses1 is not None:
        for id in range(54):
            if id in poses1:
                block_pos = poses1[id]['pos']
                block_quat = poses1[id]['orientation']
                tags_detected = poses1[id]['tags_detected']

                if calibration_mode:
                    block_pos, block_quat = swap_coordinates_cali(block_pos, block_quat)
                else:
                    block_pos, block_quat = swap_coordinates_normal(block_pos, block_quat)

                if not cali_fl:
                    block_quat = block_quat * Quaternion(axis=[1, 0, 0], degrees=180) * Quaternion(axis=[0, 1, 0], degrees=-90)
                print(f"Block#{id} quat: {repr(block_quat)}")

                sim.data.set_mocap_pos(f'block_{id}', block_pos + tag_height)
                sim.data.set_mocap_quat(f'block_{id}', block_quat.q)

                yelow = np.array([252, 186, 3, 255]) / 255
                if tags_detected == 1:
                    set_block_color(id, yelow, sim)
                else:
                    blue = np.array([66, 135, 245, 255]) / 255
                    green = np.array([50, 168, 82, 255]) / 255
                    set_block_color(id, green, sim)
                    measured_poses[id].append(poses1[id]['pos'])
            else:
                red = np.array([252, 50, 57, 255]) / 255
                set_block_color(id, red, sim)

    # if poses2 is not None:
    #     for id in range(54):
    #         if id in poses2:
    #             block_pos = poses2[id]['pos']
    #             block_quat = poses2[id]['orientation']
    #
    #             if calibration_mode:
    #                 block_pos, block_quat = swap_coordinates_cali(block_pos, block_quat)
    #             else:
    #                 block_pos, block_quat = swap_coordinates_normal(block_pos, block_quat)
    #
    #             # print(block_quat)
    #
    #             sim.data.set_mocap_pos(f'block_{54+id}', block_pos + + tag_height)
    #             sim.data.set_mocap_quat(f'block_{54+id}', block_quat.q)

def get_shild_pose_mujoco(tag_id, im, detector):
    height, width = im.shape
    camera_params_mujoco = get_camera_params_mujoco(height, width, 45)
    poses = get_tag_poses_from_image(im, [106, 107], 9.6, 224, 40, np.array([0, 0, 0]), camera_params_mujoco, detector)


    if poses is not None and tag_id in poses:
        mtx = poses[107]
        shield_quat = Quaternion(matrix=mtx)
        shield_quat = Quaternion([shield_quat[0], -shield_quat[3], shield_quat[1], -shield_quat[2]])

        temp = matrix2pose_ZYX(mtx)
        x = -temp[2]*scaler*2
        y = temp[0]*scaler*2
        z = -temp[1]*scaler*2

        shield_pose = np.array([x, y, z])

        return shield_pose, shield_quat
    else:
        return None, None

def update_shild_pos():
    while True:
        start = time.time()
        set_block_poses_debug(cam1, cam2, detector1, detector2, block_sizes, cali_fl=calibration_mode)
        stop = time.time()
        print(f"Pose estimation time: {(stop-start)*1000:.2f}ms")

        # get_shild_pose(107, cam, detector, corrections)

def take_screenshot():
    data = np.asarray(viewer.read_pixels(1920 - 66, 1080 - 55, depth=False)[::-1, :, :], dtype=np.uint8)
    data[:, :, [0, 2]] = data[:, :, [2, 0]]
    return data

def calculate_errors(sim, screenshot, tag_id):
    actual_pos = deepcopy(sim.data.get_body_xpos('shield'))
    actual_quat = sim.data.get_body_xquat('shield')
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2GRAY)

    estimated_pos, estimated_quat = get_shild_pose_mujoco(tag_id, screenshot, detector)
    if estimated_pos is not None:
        estimated_pos += cali_pos
        pos_error = np.linalg.norm(estimated_pos - actual_pos)
        quat_error = np.linalg.norm(actual_quat - estimated_quat.q)
        print(estimated_pos - actual_pos)
        print(estimated_quat - actual_quat)
        print(f"Pos error: {pos_error}")

        return estimated_pos
    else:
        return np.array([0, 0, 0])

def get_camera_params_mujoco(height, width, fovy):
    f = 0.5 * height / math.tan(fovy * math.pi / 360)
    fx = f
    fy = f
    cx = width / 2
    cy = height / 2
    return (fx, fy, cx, cy)

def swap_coordinates_cali(pos, quat):
    quat = Quaternion([quat[0], -quat[3], quat[1], -quat[2]])
    x = -pos[2] * scaler
    y = pos[0] * scaler
    z = -pos[1] * scaler

    pos = np.array([x, y, z])
    pos += cali_pos

    return pos, quat

def swap_coordinates_normal(pos, quat):
    quat = Quaternion([quat[0], quat[2], quat[1], -quat[3]])
    x = pos[1] * scaler
    y = pos[0] * scaler
    z = -pos[2] * scaler

    pos = np.array([x, y, z])

    return pos, quat

if __name__ == '__main__':

    calibration_mode = False
    target_tag_size = 9.6
    ref_tag_pos = np.array([0, 0, 0])
    if calibration_mode:
        ref_tag_id = 224
        ref_tag_size = 40
    else:
        ref_tag_id = 255
        ref_tag_size = 56.2  # big: 59.3 small: 50 fixed: 56.2

    # declare global variable
    scaler = 0.01
    cali_pos = np.array([0.0, 0.0, 25.0]) * scaler
    cali_quat = Quaternion(axis=[1, 0, 0], degrees=180)
    shield_pos = np.array([0.0, 50 + 3 + 22, -(50 - 10 - 2.8 - 12)]) * scaler
    shield_pos += cali_pos
    shield_quat = Quaternion([1, 0, 0, 0])
    measured_poses = {i: [] for i in range(54)}

    # blocks
    block_pos = np.array([0, 0, 100])
    block_quat = Quaternion([1, 0, 0, 0])

    # initialize camera and detector
    # cam = Camera(cam1_serial, cam1_mtx_11cm, cam1_dist_11cm)
    cam1 = Camera(cam1_serial, cam1_mtx, cam1_dist)
    cam2 = Camera(cam2_serial, cam2_mtx, cam2_dist)
    # cam1.start_grabbing()
    # cam2.start_grabbing()
    detector1 = dt_apriltags.Detector(nthreads=detection_threads,
                                     quad_decimate=quad_decimate1,
                                     quad_sigma=quad_sigma1,
                                     decode_sharpening=decode_sharpening1)

    detector2 = dt_apriltags.Detector(nthreads=detection_threads,
                                      quad_decimate=quad_decimate2,
                                      quad_sigma=quad_sigma2,
                                      decode_sharpening=decode_sharpening2)

    last_poses = dict()
    max_displacements = []

    # get corrections
    corrections = read_corrections('corrections.json')

    # define block sizes
    block_sizes = read_block_sizes('block_sizes.json')
    # for i in range(54, 54 + 54):
    #     block_sizes[i] = block_sizes[i-54]

    print(f"Block sizes: {block_sizes}")
    print(f"Corrections: {corrections}")



    # start thread for position update
    t = Thread(target=update_shild_pos)
    t.start()

    # initialize simulation related variables
    MODEL_XML = generate_scene_with_cali(scaler, cali_pos, shield_pos, calibration_mode)
    model = load_model_from_xml(MODEL_XML)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    viewer.cam.elevation = -25
    viewer.cam.azimuth = 180
    viewer.cam.lookat[0:3] = [-1, 0, 0]
    viewer._run_speed = 16
    viewer.cam.distance = 3
    t = 0


    estimated_shield_pos = np.array([0, 0, 0])

    # simulate
    while t < 100000:
        t += 1

        if t % 100 == 0:
            pass

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

    print(f"Max displacements: {max_displacements}")
    print(f"Max displacements mean: {np.mean(max_displacements)}")

    exit()

    dataset = []
    for id in measured_poses:
        mean = np.mean(measured_poses[id], axis=0)
        print(f"Mean = {mean}")
        errs = dict()
        for m in measured_poses[id]:
            err = m - mean
            data = {'id': id, 'err': err[0], 'axis': 'x'}
            dataset.append(data)
            data = {'id': id, 'err': err[1], 'axis': 'y'}
            dataset.append(data)
            data = {'id': id, 'err': err[2], 'axis': 'z'}
            dataset.append(data)

    df = pd.DataFrame(dataset)
    sns.stripplot(x='id', y='err', hue='axis', data=df, dodge=True)
    plt.show()
