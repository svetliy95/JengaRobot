import numpy as np
from constants import *
from pyquaternion import Quaternion
from scipy.stats import truncnorm
from cv.block_localization import get_block_positions_mujoco, get_camera_params_mujoco, get_block_positions
import cv2
import math
from utils.utils import *
import time
import logging
import colorlog
import copy
import random
# from mujoco_py import MjRenderContextOffscreen

# specify logger
# DEBUG: Detailed information, typically of interest only when diagnosing problems.
# INFO: Confirmation that things are working as expected.
# WARNING: An indication that something unexpected happened, or indicative of some problem in
# the near future (e.g. ‘disk space low’). The software is still working as expected.
# ERROR: Due to a more serious problem, the software has not been able to perform some function.
# CRITICAL: A serious error, indicating that the program itself may be unable to continue running.

log = logging.Logger(__name__)
formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(funcName)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.INFO)
log.addHandler(stream_handler)

# log = logging.Logger(__name__)
# file_formatter = logging.Formatter('%(asctime)s:%(levelname)sMain process:PID:%(process)d:%(funcName)s:%(message)s')
# file_handler = logging.FileHandler(filename='tower.log', mode='w')
# file_handler.setFormatter(file_formatter)
# file_handler.setLevel(logging.DEBUG)
# log.addHandler(file_handler)


class Tower:
    pos = origin
    block_num = 54
    block_prefix = "block"
    sim = None
    ref_positions = {}
    last_ref_positions = {}
    ref_orientations = {}
    last_ref_orientations = {}
    block_sizes = []
    placing_vert_spacing = block_height_max / 3
    placing_horiz_spacing = one_millimeter
    pertrubation = False
    n_sigma = 2
    position_error = 4 * one_millimeter  # error within n_sigma
    pos_sigma = (position_error / n_sigma) / math.sqrt(3)
    orientation_sigma = 2/3

    def __init__(self, sim=None, viewer=None, simulation_fl=True, cam1=None, cam2=None, at_detector=None, block_sizes=None, corrections=None):

        if simulation_fl:
            log.debug(f"Init #1")
            self.sim = sim
            self.viewer = viewer
            positions = self.get_positions()
            orientations = self.get_orientations()
            log.debug(f"Positions: {positions}")
            log.debug(f"Orientations: {orientations}")
            self.ref_positions = copy.deepcopy(positions)
            self.last_ref_positions = copy.deepcopy(positions)
            self.ref_orientations = copy.deepcopy(orientations)
            self.last_ref_orientations = copy.deepcopy(orientations)
            self.toppled_fl = False
            log.debug(f"Init #2")
        else:

            # debigging
            self.image_index = 0
            ######################

            self.cam1 = cam1
            self.cam2 = cam2
            self.at_detector = at_detector
            self.block_sizes = block_sizes
            self.corrections = corrections
            self.target_tag_size = 9.6
            self.ref_tag_pos = np.array([0, 0, 0])
            self.ref_tag_id = 255
            self.ref_tag_size = 56.2

            # initialize flipping corrections
            self.orientation_corrections = {i: Quaternion([1, 0, 0, 0]) for i in range(g_blocks_max)}

            poses = self.get_poses_cv()
            positions = self.get_positions_from_poses(poses)
            orientations = self.get_orientations_from_poses(poses)

            # calculate corrections for the orientations
            self.calculate_orientation_corrections(orientations)

            # apply corrections
            for i in range(g_blocks_max):
                if i in orientations:
                    orientations[i] *= self.orientation_corrections[i]

            self.ref_positions = copy.deepcopy(positions)
            self.last_ref_positions = copy.deepcopy(positions)
            self.ref_orientations = copy.deepcopy(orientations)
            self.last_ref_orientations = copy.deepcopy(orientations)
            self.toppled_fl = False
            self.initial_pos = self.get_center_xy(positions)


    def reset_orientations(self, orientations):
        # calculate corrections for the orientations
        self.calculate_orientation_corrections(orientations)

    # calculates corrections needed when blocks are flipped over or rotated around z-axis
    def calculate_orientation_corrections(self, orientations):
        for i in range(g_blocks_max):
            correction = self.orientation_corrections[i]
            if i in orientations:
                q = orientations[i]
                angle_x = angle_between_vectors(x_unit_vector, q.rotate(x_unit_vector))
                angle_x = math.degrees(angle_x)
                angle_y = angle_between_vectors(y_unit_vector, q.rotate(x_unit_vector))  # this should be x_unit_vector, not y_unit_vector
                angle_y = math.degrees(angle_y)
                angle_z = angle_between_vectors(z_unit_vector, q.rotate(z_unit_vector))
                angle_z = math.degrees(angle_z)

                print(f"Block #{i}: X: {angle_x}, Y: {angle_y}, Z: {angle_z}")

                if abs(angle_x - 180) < flipping_threshold or abs(angle_y - 180) < flipping_threshold:
                    correction *= Quaternion(axis=z_unit_vector, degrees=180)

                if abs(angle_z - 180) < flipping_threshold:
                    correction *= Quaternion(axis=x_unit_vector, degrees=180)

            self.orientation_corrections[i] = correction

    def get_position(self, num):
        assert num < self.block_num, "Block num is to high"
        # copy the numpy array with the position, otherwise the value in the variable may change
        exact_pos = copy.deepcopy(self.sim.data.get_body_xpos(self.block_prefix + str(num)))
        if Tower.pertrubation:
            start = time.time()
            distorted_pos = random.gauss(exact_pos, self.pos_sigma)
            elapsed = time.time() - start
            # log.info(f"Position elapsed: {elapsed * 1000:.2f}ms")
            # log.info(f"Position error: {np.linalg.norm(distorted_pos - exact_pos)/one_millimeter}")
            return distorted_pos
        else:
            return exact_pos

    def get_orientation(self, num):
        assert num < self.block_num, "Block num is to high"
        exact_orientation = Quaternion(self.sim.data.get_body_xquat(self.block_prefix + str(num)))

        if Tower.pertrubation:
            start = time.time()
            q_distorted = exact_orientation * \
                          Quaternion(axis=[1, 0, 0], degrees=random.gauss(0, self.orientation_sigma)) * \
                          Quaternion(axis=[0, 1, 0], degrees=random.gauss(0, self.orientation_sigma)) * \
                          Quaternion(axis=[0, 0, 1], degrees=random.gauss(0, self.orientation_sigma))
            elapsed = time.time() - start
            # log.info(f"Orinetation elapsed: {elapsed*1000:.2f}ms")
            # log.info(f"Orientation error: {math.degrees(get_angle_between_quaternions(q_distorted, exact_orientation))}")
            return q_distorted.q
        else:
            return exact_orientation.q

    def _take_picture(self, cam_id):
        w = 4096
        h = 2160
        if isinstance(self.viewer, MjRenderContextOffscreen):
            self.viewer.render(w, h, cam_id)
        else:  # on-screen rendering
            # switch to fixed cam
            self.viewer.cam.fixedcamid = cam_id
        data = np.asarray(self.viewer.read_pixels(w, h, depth=False)[::-1, :, :], dtype=np.uint8)
        data[:, :, [0, 2]] = data[:, :, [2, 0]]
        return data

    @staticmethod
    def _get_angle_between_quaternions(q1, q2):
        q1 = Quaternion(q1)
        q2 = Quaternion(q2)
        v1 = q1.rotate(x_unit_vector)
        v2 = q2.rotate(x_unit_vector)
        return angle_between_vectors(v1, v2)

    def get_poses_cv_mujoco(self, ids=[], return_images=True):
        assert set(ids).issubset(set(range(g_blocks_num))), "Wrong block ids!"
        if not ids:
            ids = [i for i in range(g_blocks_num)]

        im1 = self._take_picture(0)
        im2 = self._take_picture(1)
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        height, width = im1_gray.shape

        poses, dimage1, dimage2 = get_block_positions_mujoco(im1=im1_gray,
                                                             im2=im2_gray,
                                                             block_ids=ids,
                                                             target_tag_size=block_height_mean * 0.8,  # 0.8 because the actual tag length is 8/10 of the whole length
                                                             ref_tag_size=coordinate_frame_tag_size[0] * 0.8,  # 0.8 because the actual tag length is 8/10 of the whole length
                                                             ref_tag_pos=coordinate_frame_tag_pos,
                                                             ref_tag_id=coordinate_frame_tag_id,
                                                             block_sizes=np.array([block_length_mean, block_width_mean, block_height_mean]),
                                                             camera_params=get_camera_params_mujoco(height, width, fovy),
                                                             return_images=return_images)

        # calculate errors
        for i in poses:
            poses[i]['pos_error'] = np.linalg.norm(poses[i]['pos'] - self.get_position(i))
            poses[i]['orientation_error'] = math.degrees(Tower._get_angle_between_quaternions(poses[i]['orientation'], self.get_orientation(i)))

        # mark detected tags on images
        white_areas = (dimage1 == 255)
        im1[..., ][white_areas] = (100, 255, 100)
        white_areas = (dimage2 == 255)
        im2[..., ][white_areas] = (100, 255, 100)

        if return_images:
            return poses, im1, im2
        else:
            return poses

    def get_poses_cv(self):
        cam_params1 = self.cam1.get_params()
        cam_params2 = self.cam2.get_params()
        im1 = self.cam1.get_raw_image()
        im2 = self.cam2.get_raw_image()
        cv2.imwrite(f'./debug_images/image_{self.image_index}.jpg', im1)
        self.image_index += 1
        cv2.imwrite(f'./debug_images/image_{self.image_index}.jpg', im2)
        self.image_index += 1
        block_ids = [i for i in range(54)]
        poses = get_block_positions(im1, im2, block_ids, self.target_tag_size, self.ref_tag_size, self.ref_tag_id, self.ref_tag_pos,
                                     self.block_sizes, self.corrections,
                                     cam_params1, cam_params2, False, self.at_detector, cam1_mtx, cam1_dist, cam2_mtx,
                                     cam2_dist)

        if poses is not None:
            for id in poses:
                block_pos = poses[id]['pos']
                block_quat = poses[id]['orientation']

                block_pos, block_quat = self.swap_coordinates(block_pos, block_quat)
                block_quat = block_quat * Quaternion(axis=[1, 0, 0], degrees=180) * Quaternion(axis=[0, 1, 0],
                                                                                               degrees=-90)
                poses[id]['pos'] = block_pos
                poses[id]['orientation'] = block_quat * self.orientation_corrections[id]


        return poses

    def swap_coordinates(self, pos, quat):
        quat = Quaternion([quat[0], quat[2], quat[1], -quat[3]])
        x = pos[1]
        y = pos[0]
        z = -pos[2]

        pos = np.array([x, y, z])

        return pos, quat

    def get_position_cv(self, num):
        assert num < self.block_num, "Block num is to high"

        position = self.get_poses_cv([num])

        if len(position) > 0:
            return self.get_poses_cv([num])[0]
        else:
            return None

    def get_positions_from_poses(self, poses):
        positions = {id: poses[id]['pos'] for id in poses}
        return positions

    def get_orientations_from_poses(self, poses):
        positions = {id: poses[id]['orientation'] for id in poses}
        return positions

    def get_highest_block_id(self, positions, current_block):
        max_z = 0
        max_block_num = -1
        for i in positions:
            if i != current_block:
                height = positions[i][2]
                if height > max_z:
                    max_z = height
                    max_block_num = i

        return max_block_num

    def get_lowest_block_id(self, positions):
        min_z = np.Inf
        min_block_id = -1
        for i in positions:
            height = positions[i][2]
            if height < min_z:
                min_z = height
                min_block_id = i

        return min_block_id

    def get_blocks_from_highest_level(self, positions):
        highest_block = self.get_highest_block_id(positions)
        highest_block_pos = self.get_position(highest_block)

    def get_positions(self):
        positions = {}
        for i in range(g_blocks_num):
            positions[i] = self.get_position(i)

        return positions

    def get_orientations(self):
        orientations = {}
        for i in range(g_blocks_num):
            orientations[i] = self.get_orientation(i)

        return orientations

    def get_adjacent_blocks(self, id, positions):
        root_pos = positions[id]

        # search for blocks with the same height within a certain threshold
        adjacent_blocks = []
        for i in positions:
            if abs(positions[i][2] - root_pos[2]) < same_height_threshold:
                adjacent_blocks.append(i)

        log.debug(f"Adjacent blocks: {adjacent_blocks}")

        return adjacent_blocks

    # remove blocks that are far from the tower
    def filter_positions(self, positions):
        # TODO: can be implemented more efficiently
        ids = list(positions.keys())
        start = time.time()
        blocks_within_tower = set()
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                dist = np.linalg.norm(positions[ids[i]] - positions[ids[i]])
                # log.debug(f"Blocks: [{i}] and [{j}], dist: {dist}, threshold: {block_to_far_threshold}")
                if dist < block_to_far_threshold:
                    blocks_within_tower.update([ids[i], ids[j]])
                    break
        elapsed = time.time() - start
        log.debug(f"Elapsed: {elapsed * 1000:.2f}ms")
        log.debug(f"Ignored blocks: {set(positions.keys()) - blocks_within_tower}")

        return {i: positions[i] for i in blocks_within_tower}

    def get_layers(self, positions):
        # filter out blocks that are too far from the tower and therefore are not a part of the tower
        positions = self.filter_positions(positions)

        # sort blocks based on z-coordinate
        positions = {k: v for k, v in sorted(positions.items(), key=lambda x: x[1][2])}

        dict_keys = list(positions.keys())
        layers = []
        i = 0
        while i < len(dict_keys):
            layer = [dict_keys[i]]
            while (i < len(dict_keys) - 1) and abs(positions[dict_keys[i]][2] - positions[dict_keys[i+1]][2]) < same_height_threshold:
                layer.append(dict_keys[i+1])
                i += 1
            i += 1
            layers.append(layer)
        return layers

    def get_layers_old(self, positions):
        blocks_to_test = [i for i in range(self.block_num)]
        layers = []
        for i in range(self.block_num):
            layer = []
            if i in blocks_to_test:
                for j in range(self.block_num):
                    if j in blocks_to_test:
                        if abs(positions[i][2] - positions[j][2]) < same_height_threshold:
                            layer.append(j)
                layers.append(layer)
                blocks_to_test = (set(blocks_to_test) - set(layer))

        # sort list of layers based on the height
        layers.sort(key=lambda x: positions[x[0]][2])

        return layers

    def mean_height(self, layer, positions):
        heights = []
        for id in layer:
            heights.append(positions[id][2])
        return np.mean(heights)

    def mean_orientation(self, layer, orientations):
        layer_orientations = []
        for block_id in layer:
            layer_orientations.append(orientations[block_id])
        mean_orientation = average_quaternions(layer_orientations)

        return mean_orientation

    def get_possible_positions(self, layer, positions, orientations, origin):
        mean_layer_height = self.mean_height(layer, positions)

        # calculate mean orientation of the blocks on the highest full layer
        mean_orientation = self.mean_orientation(layer, orientations)

        # log.debug(f"Mean orientation: {mean_orientation}")
        # log.debug(f"Mean orientation ypr: {mean_orientation.yaw_pitch_roll}")

        # calculate center of the tower
        tower_center = self.get_center_xy(positions)
        tower_center_with_height = tower_center
        tower_center_with_height[2] = mean_layer_height

        # calculate centers for each position
        offset_orientation = mean_orientation * Quaternion(axis=z_unit_vector, degrees=90)

        # print(f"Offset_orientation: {offset_orientation.yaw_pitch_roll}")

        offset_vector = offset_orientation.rotate(x_unit_vector)
        offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                     p=tower_center_with_height,
                                                                     origin=origin)
        pos1 = tower_center_with_height + (block_width_max * offset_direction)
        pos2 = tower_center_with_height
        pos3 = tower_center_with_height - (block_width_max * offset_direction)
        possible_positions = [pos1, pos2, pos3]

        return possible_positions

    def get_layers_state(self, positions, orientations, origin):
        layers = self.get_layers(positions)
        layers_state = {}

        print(f"Layers: {layers}")

        for i in range(len(layers)):


            possible_positions = self.get_possible_positions(layers[i], positions, orientations, origin)
            occupied_positions = self._assign_pos(layers[i], possible_positions, positions)

            layers_state[i] = {0: occupied_positions[0] if 0 in occupied_positions else None,
                               1: occupied_positions[1] if 1 in occupied_positions else None,
                               2: occupied_positions[2] if 2 in occupied_positions else None}
        return layers_state

    def get_full_layers(self, positions):
        layers = self.get_layers(positions)

        full_layers = [layer for layer in layers if len(layer) == 3]

        return full_layers

    def get_center_xy(self, positions):
        full_layers = self.get_full_layers(positions)
        log.debug(f"Full layers: {full_layers}")
        x = []
        y = []
        for layer in full_layers:
            for id in layer:
                x.append(positions[id][0])
                y.append(positions[id][1])
        log.debug(f"Tower center: {np.array([np.mean(x), np.mean(y)])}")
        return np.array([np.mean(x), np.mean(y), 0])

    def get_highest_layer(self, positions, current_block):
        return self.get_adjacent_blocks(self.get_highest_block_id(positions, current_block), positions)

    def get_highest_full_layer(self, positions):
        max_z = -np.Inf
        highest_layer = None
        full_layers = self.get_full_layers(positions)
        for layer in full_layers:
            z = positions[layer[0]][2]
            if z > max_z:
                highest_layer = layer
                max_z = z

        return highest_layer

    @staticmethod
    def _assign_pos(layer, possible_positions, positions):
        distances = {}
        for block in layer:
            distances_for_block = []
            for pos in possible_positions:
                distances_for_block.append(np.linalg.norm(pos - positions[block]))

            distances[block] = distances_for_block

        # log.debug(f"Possible positions: {possible_positions}")
        # log.debug(f"Distances: {distances}")
        # for i in layer:
            # log.debug(f"Position of block #{i}: {positions[i]}")

        occupied_positions = {}
        for block in layer:
            # log.debug(f"Current block: {block}")
            # log.debug(f"Distances for current block: {distances}")
            # log.debug(f"Argmin for current block: {np.argmin(distances[block])}")
            occupied_positions[np.argmin(distances[block])] = block

        # log.debug(f"Occupied positions: {occupied_positions}")

        return occupied_positions

    def get_placing_pose_mujoco(self, positions, orientations, current_block):
        # There are 3 cases:
        # 1. 3 blocks in the highest layer
        # 2. 2 blocks in the highest layer
        # 3. 1 block in the highest layer
        # And three possible positions for blocks
        # Position 1: offset in the direction of the origin
        # Position 2: central position
        # Position 3: offset in the direction away from the origin


        try:
            # calculate highest full layer mean height
            highest_full_layer = self.get_highest_full_layer(positions)
            z = []
            for i in range(3):
                z.append(positions[highest_full_layer[i]][2])
            highest_full_layer_height = np.mean(z)

            # calculate mean orientation of the blocks on the highest full layer
            layer_orientations = []
            for block_id in highest_full_layer:
                layer_orientations.append(orientations[block_id])
            layer_orientations = np.array(layer_orientations)
            mean_orientation = Quaternion(np.mean(layer_orientations, axis=0))  # get the mean orientation of the top blocks

            log.debug(f"Orientations: {layer_orientations}")
            log.debug(f"Mean orientation: {mean_orientation}")
            log.debug(f"Mean orientation ypr: {mean_orientation.yaw_pitch_roll}")

            # calculate center of the tower
            tower_center = self.get_center_xy(positions)
            tower_center_with_height = np.concatenate(
                (tower_center, np.array([highest_full_layer_height + block_height_max])))

            # calculate centers for each position
            offset_orientation = mean_orientation

            print(f"Offset_orientation: {offset_orientation.yaw_pitch_roll}")

            offset_vector = offset_orientation.rotate(x_unit_vector)
            offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                         p=tower_center_with_height,
                                                                         origin=np.array([coordinate_axes_pos_x,
                                                                                          coordinate_axes_pos_y,
                                                                                          coordinate_axes_pos_z]))
            pos1 = tower_center_with_height + (block_width_max * offset_direction)
            pos2 = tower_center_with_height
            pos3 = tower_center_with_height - (block_width_max * offset_direction)
            possible_positions = [pos1, pos2, pos3]

            # get the highest (not necessary full) layer
            highest_layer = self.get_highest_layer(positions, current_block)

            # get occupied positions
            occupied_positions = self._assign_pos(highest_layer, possible_positions, positions)

            # get stopover
            stopover = tower_center_with_height + \
                       offset_direction * (block_width_max * 3) + \
                       Tower.placing_vert_spacing
            log.debug(f"Stopover: {stopover}")

            # case 1: place the block on the side near the origin perpendicular to the last 3 blocks
            if len(highest_layer) == 3:
                # block orientation must be perpendicular to the top blocks
                block_orientation = mean_orientation * Quaternion(axis=[0, 0, 1], degrees=-90)

                # calculate pos
                temp_vector = mean_orientation.rotate(x_unit_vector)
                offset_direction = get_direction_towards_origin_along_vector(vec=temp_vector,
                                                                             p=tower_center_with_height,
                                                                             origin=np.array([coordinate_axes_pos_x,
                                                                                              coordinate_axes_pos_y,
                                                                                              coordinate_axes_pos_z]))

                placing_pos = tower_center_with_height - block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                placing_pos_with_tolerance = placing_pos + z_unit_vector * Tower.placing_vert_spacing

            # case 2: we consider only the cases where the two blocks are lying near each other
            # the case where the blocks are lying on the sides of the tower is not considered
            if len(highest_layer) == 2:

                # find position, where to place
                if 0 in occupied_positions and 1 in occupied_positions:
                    central_block_position = positions[occupied_positions[1]]
                    central_block_orientation = Quaternion(self.get_orientation(occupied_positions[1]))
                    offset_orientation = central_block_orientation * Quaternion(axis=[0, 0, 1], degrees=90)
                    offset_vector = offset_orientation.rotate(x_unit_vector)
                    offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                             p=central_block_position,
                                                                             origin=np.array([coordinate_axes_pos_x,
                                                                                              coordinate_axes_pos_y,
                                                                                              coordinate_axes_pos_z]))

                    placing_pos = central_block_position - block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                    placing_pos_with_tolerance = placing_pos + offset_direction * Tower.placing_horiz_spacing + z_unit_vector * Tower.placing_vert_spacing

                    block_orientation = central_block_orientation

                if 1 in occupied_positions and 2 in occupied_positions:
                    central_block_position = positions[occupied_positions[1]]
                    central_block_orientation = Quaternion(self.get_orientation(occupied_positions[1]))
                    offset_orientation = central_block_orientation * Quaternion(axis=[0, 0, 1], degrees=90)
                    offset_vector = offset_orientation.rotate(x_unit_vector)
                    offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                                 p=central_block_position,
                                                                                 origin=np.array([coordinate_axes_pos_x,
                                                                                                  coordinate_axes_pos_y,
                                                                                                  coordinate_axes_pos_z]))
                    placing_pos = central_block_position + block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                    placing_pos_with_tolerance = placing_pos + offset_direction * Tower.placing_horiz_spacing + z_unit_vector * Tower.placing_vert_spacing

                    block_orientation = central_block_orientation

                if 0 in occupied_positions and 2 in occupied_positions:
                    log.error("Stupid blocks placing!")

            # Case 3
            if len(highest_layer) == 1:
                if 0 in occupied_positions:
                    block0_pos = positions[occupied_positions[0]]
                    block0_orientation = Quaternion(self.get_orientation(occupied_positions[0]))
                    offset_orientation = block0_orientation * Quaternion(axis=[0, 0, 1], degrees=90)
                    offset_vector = offset_orientation.rotate(x_unit_vector)
                    offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                                 p=block0_pos,
                                                                                 origin=np.array([coordinate_axes_pos_x,
                                                                                                  coordinate_axes_pos_y,
                                                                                                  coordinate_axes_pos_z]))

                    placing_pos = block0_pos - block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                    placing_pos_with_tolerance = placing_pos + offset_direction * Tower.placing_horiz_spacing + z_unit_vector * Tower.placing_vert_spacing

                    block_orientation = block0_orientation

                if 1 in occupied_positions:
                    block1_pos = positions[occupied_positions[1]]
                    block1_orientation = Quaternion(self.get_orientation(occupied_positions[1]))
                    offset_orientation = block1_orientation * Quaternion(axis=[0, 0, 1], degrees=90)
                    offset_vector = offset_orientation.rotate(x_unit_vector)
                    offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                                 p=block1_pos,
                                                                                 origin=np.array([coordinate_axes_pos_x,
                                                                                                  coordinate_axes_pos_y,
                                                                                                  coordinate_axes_pos_z]))

                    placing_pos = block1_pos + block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                    placing_pos_with_tolerance = placing_pos + offset_direction * Tower.placing_horiz_spacing + z_unit_vector * Tower.placing_vert_spacing

                    block_orientation = block1_orientation

                if 2 in occupied_positions:
                    block2_pos = positions[occupied_positions[2]]
                    block2_orientation = Quaternion(self.get_orientation(occupied_positions[2]))
                    offset_orientation = block2_orientation * Quaternion(axis=[0, 0, 1], degrees=90)
                    offset_vector = offset_orientation.rotate(x_unit_vector)
                    offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                                 p=block2_pos,
                                                                                 origin=np.array([coordinate_axes_pos_x,
                                                                                                  coordinate_axes_pos_y,
                                                                                                  coordinate_axes_pos_z]))

                    placing_pos = block2_pos + block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                    placing_pos_with_tolerance = placing_pos + offset_direction * Tower.placing_horiz_spacing + z_unit_vector * Tower.placing_vert_spacing

                    block_orientation = block2_orientation

            num_of_blocks = len(highest_layer)

            log.debug(f"Occupied positions: {occupied_positions}")
            log.debug(f"Highest layer: {highest_layer}")
            log.debug(f"Highest full layer: {highest_full_layer}")
            log.debug(f"Placing pos: {placing_pos}")

            return {'pos': placing_pos,
                    'pos_with_tolerance': placing_pos_with_tolerance,
                    'orientation': block_orientation,
                    'stopover': stopover,
                    'num_of_blocks': num_of_blocks}
        except:
            log.exception(f"Exception!")
            log.debug(f"Privet!")
            self.toppled_fl = True
            return {'pos': np.array([0, 0, 0]),
                    'pos_with_tolerance': np.array([0, 0, 0]),
                    'orientation': Quaternion([1, 0, 0, 0]),
                    'stopover': np.array([0, 0, 0]),
                    'num_of_blocks': 0}

    def get_placing_pose(self, positions, orientations, current_block):
        # There are 3 cases:
        # 1. 3 blocks in the highest layer
        # 2. 2 blocks in the highest layer
        # 3. 1 block in the highest layer
        # And three possible positions for blocks
        # Position 1: offset in the direction of the origin
        # Position 2: central position
        # Position 3: offset in the direction away from the origin


        try:
            # calculate highest full layer mean height
            highest_full_layer = self.get_highest_full_layer(positions)
            z = []
            for i in range(3):
                z.append(positions[highest_full_layer[i]][2])
            highest_full_layer_height = np.mean(z)

            # calculate mean orientation of the blocks on the highest full layer
            layer_orientations = []
            for block_id in highest_full_layer:
                layer_orientations.append(orientations[block_id])
            layer_orientations = np.array(layer_orientations)
            mean_orientation = Quaternion(np.mean(layer_orientations, axis=0))  # get the mean orientation of the top blocks

            log.debug(f"Orientations: {layer_orientations}")
            log.debug(f"Mean orientation: {mean_orientation}")
            log.debug(f"Mean orientation ypr: {mean_orientation.yaw_pitch_roll}")

            # calculate center of the tower
            tower_center = self.get_center_xy(positions)
            tower_center_with_height = tower_center
            tower_center_with_height[2] = highest_full_layer_height + block_height_max


            # calculate centers for each position
            offset_orientation = mean_orientation

            print(f"Offset_orientation: {offset_orientation.yaw_pitch_roll}")

            offset_vector = offset_orientation.rotate(x_unit_vector)
            offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                         p=tower_center_with_height,
                                                                         origin=origin)
            pos1 = tower_center_with_height + (block_width_max * offset_direction)
            pos2 = tower_center_with_height
            pos3 = tower_center_with_height - (block_width_max * offset_direction)
            possible_positions = [pos1, pos2, pos3]

            # get the highest (not necessary full) layer
            highest_layer = self.get_highest_layer(positions, current_block)

            # get occupied positions
            occupied_positions = self._assign_pos(highest_layer, possible_positions, positions)

            # get stopover
            stopover = tower_center_with_height - \
                       offset_direction * (block_width_max * 3) + \
                       Tower.placing_vert_spacing
            log.debug(f"Stopover: {stopover}")

            # case 1: place the block on the side near the origin perpendicular to the last 3 blocks
            if len(highest_layer) == 3:
                # block orientation must be perpendicular to the top blocks
                block_orientation = mean_orientation * Quaternion(axis=[0, 0, 1], degrees=90)

                # calculate pos
                temp_vector = mean_orientation.rotate(x_unit_vector)
                offset_direction = get_direction_towards_origin_along_vector(vec=temp_vector,
                                                                             p=tower_center_with_height,
                                                                             origin=origin)

                placing_pos = tower_center_with_height + block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                placing_pos_with_tolerance = placing_pos + z_unit_vector * Tower.placing_vert_spacing

            # case 2: we consider only the cases where the two blocks are lying near each other
            # the case where the blocks are lying on the sides of the tower is not considered
            if len(highest_layer) == 2:

                # find position, where to place
                if 0 in occupied_positions and 1 in occupied_positions:
                    central_block_position = positions[occupied_positions[1]]
                    central_block_orientation = Quaternion(orientations[occupied_positions[1]])
                    offset_orientation = central_block_orientation * Quaternion(axis=[0, 0, 1], degrees=90)
                    offset_vector = offset_orientation.rotate(x_unit_vector)
                    offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                             p=central_block_position,
                                                                             origin=origin)

                    placing_pos = central_block_position - block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                    placing_pos_with_tolerance = placing_pos + offset_direction * Tower.placing_horiz_spacing + z_unit_vector * Tower.placing_vert_spacing

                    block_orientation = central_block_orientation

                if 1 in occupied_positions and 2 in occupied_positions:
                    central_block_position = positions[occupied_positions[1]]
                    central_block_orientation = Quaternion(orientations[occupied_positions[1]])
                    offset_orientation = central_block_orientation * Quaternion(axis=[0, 0, 1], degrees=90)
                    offset_vector = offset_orientation.rotate(x_unit_vector)
                    offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                                 p=central_block_position,
                                                                                 origin=origin)
                    placing_pos = central_block_position + block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                    placing_pos_with_tolerance = placing_pos + offset_direction * Tower.placing_horiz_spacing + z_unit_vector * Tower.placing_vert_spacing

                    block_orientation = central_block_orientation

                if 0 in occupied_positions and 2 in occupied_positions:
                    log.error("Stupid blocks placing!")

            # Case 3
            if len(highest_layer) == 1:
                if 0 in occupied_positions:
                    block0_pos = positions[occupied_positions[0]]
                    block0_orientation = Quaternion(orientations[occupied_positions[0]])
                    offset_orientation = block0_orientation * Quaternion(axis=[0, 0, 1], degrees=90)
                    offset_vector = offset_orientation.rotate(x_unit_vector)
                    offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                                 p=block0_pos,
                                                                                 origin=origin)

                    placing_pos = block0_pos - block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                    placing_pos_with_tolerance = placing_pos + offset_direction * Tower.placing_horiz_spacing + z_unit_vector * Tower.placing_vert_spacing

                    block_orientation = block0_orientation

                if 1 in occupied_positions:
                    block1_pos = positions[occupied_positions[1]]
                    block1_orientation = Quaternion(orientations[occupied_positions[1]])
                    offset_orientation = block1_orientation * Quaternion(axis=[0, 0, 1], degrees=90)
                    offset_vector = offset_orientation.rotate(x_unit_vector)
                    offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                                 p=block1_pos,
                                                                                 origin=origin)

                    placing_pos = block1_pos + block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                    placing_pos_with_tolerance = placing_pos + offset_direction * Tower.placing_horiz_spacing + z_unit_vector * Tower.placing_vert_spacing

                    block_orientation = block1_orientation

                if 2 in occupied_positions:
                    block2_pos = positions[occupied_positions[2]]
                    block2_orientation = Quaternion(orientations[occupied_positions[2]])
                    offset_orientation = block2_orientation * Quaternion(axis=[0, 0, 1], degrees=90)
                    offset_vector = offset_orientation.rotate(x_unit_vector)
                    offset_direction = get_direction_towards_origin_along_vector(vec=offset_vector,
                                                                                 p=block2_pos,
                                                                                 origin=origin)

                    placing_pos = block2_pos + block_width_max * offset_direction + z_unit_vector * Tower.placing_vert_spacing
                    placing_pos_with_tolerance = placing_pos + offset_direction * Tower.placing_horiz_spacing + z_unit_vector * Tower.placing_vert_spacing

                    block_orientation = block2_orientation

            num_of_blocks = len(highest_layer)

            log.debug(f"Occupied positions: {occupied_positions}")
            log.debug(f"Highest layer: {highest_layer}")
            log.debug(f"Highest full layer: {highest_full_layer}")
            log.debug(f"Placing pos: {placing_pos}")

            return {'pos': placing_pos,
                    'pos_with_tolerance': placing_pos_with_tolerance,
                    'orientation': block_orientation,
                    'stopover': stopover,
                    'num_of_blocks': num_of_blocks}
        except:
            log.exception(f"Exception!")
            log.debug(f"Privet!")
            self.toppled_fl = True
            return {'pos': right_robot_home_position_world[:3],
                    'pos_with_tolerance': right_robot_home_position_world[:3],
                    'orientation': Quaternion([1, 0, 0, 0]),
                    'stopover': right_robot_home_position_world[:3],
                    'num_of_blocks': 0}

    def get_angle_to_ground(self, num):
        q = Quaternion(self.get_orientation(num))
        block_z_normal = q.rotate(z_unit_vector)

        return np.rad2deg(np.arccos(np.dot(z_unit_vector, block_z_normal)/(np.linalg.norm(block_z_normal))))

    def get_mean_angle_to_ground(self):
        angles = []
        for i in range(self.block_num):
            angles.append(self.get_angle_to_ground(i))

        return np.mean(angles)

    def get_angle_of_highest_block_to_ground(self, positions):
        num = self.get_highest_block_id(positions)
        return self.get_angle_to_ground(num)

    def get_lowest_full_layer(self, positions):
        min_z = np.Inf
        lowest_layer = None
        full_layers = self.get_full_layers(positions)
        for layer in full_layers:
            z = self.get_position(layer[0])[2]
            if z < min_z:
                lowest_layer = layer
                min_z = z

        return lowest_layer

    def get_lowest_layer(self, positions):
        return self.get_adjacent_blocks(self.get_lowest_block_id(positions), positions)

    def mean_vector(self, list_of_vectors):
        arr = np.reshape(list_of_vectors, (len(list_of_vectors), 3))
        mean_vector = np.mean(arr, axis=0)
        return mean_vector

    def mean_vector_old(self, list_of_vectors):
        new_list = []
        for v in list_of_vectors:
            new_list.append(np.reshape(v, (1, 3)))
        return np.reshape(np.mean(new_list, axis=0), 3)

    def toppled(self, positions, current_block):
        counter = 0
        for i in range(g_blocks_num):
            if i != current_block:
                distance_vector = positions[i][0:2] - self.pos[0:2]
                distance = np.linalg.norm(distance_vector)
                if distance >= toppled_distance:
                    counter += 1

        if counter > toppled_block_threshold or self.toppled_fl:
            return True
        else:
            return False

    def get_tilt_1ax(self, positions):
        # calculate 2 tilt directions
        lowest_layer = self.get_lowest_layer(positions)
        x = []
        y = []
        for id in lowest_layer:
            orientation = Quaternion(self.get_orientation(id))
            x.append(orientation.rotate(x_unit_vector))
            y.append(orientation.rotate(y_unit_vector))
        x_axis_base = self.mean_vector(x)
        y_axis_base = self.mean_vector(y)
        base_normal = np.cross(x_axis_base, y_axis_base)

        highest_layer = self.get_highest_full_layer(positions)

        # if there is no highest full layer, then return zero
        if highest_layer is None:
            return 0

        x = []
        y = []
        for id in highest_layer:
            orientation = Quaternion(self.get_orientation(id))
            x.append(orientation.rotate(x_unit_vector))
            y.append(orientation.rotate(y_unit_vector))
        x_axis_layer = self.mean_vector(x)
        y_axis_layer = self.mean_vector(y)
        normal = np.cross(x_axis_layer, y_axis_layer)


        angle = math.degrees(angle_between_vectors(normal, base_normal))
        log.debug(f"Angle: {angle}")

        return angle

    def get_tilt_2ax(self, positions, orientations, current_block):
        # calculate 2 tilt directions
        lowest_layer = self.get_lowest_layer(positions)
        x = []
        y = []
        for id in lowest_layer:
            orientation = Quaternion(orientations[id])
            x.append(orientation.rotate(x_unit_vector))
            y.append(orientation.rotate(y_unit_vector))
        x_axis_base = self.mean_vector(x)
        y_axis_base = self.mean_vector(y)
        # log.debug(f"X base: {x_axis_base}")
        # log.debug(f"Y base: {y_axis_base}")

        highest_layer = self.get_highest_layer(positions, current_block)
        x = []
        y = []

        for id in highest_layer:
            orientation = Quaternion(orientations[id])
            x.append(orientation.rotate(x_unit_vector))
            y.append(orientation.rotate(y_unit_vector))

        x_axis_layer = self.mean_vector(x)
        y_axis_layer = self.mean_vector(y)
        normal = np.cross(x_axis_layer, y_axis_layer)
        # log.debug(f"X layer: {x_axis_layer}")
        # log.debug(f"Y layer: {y_axis_layer}")
        # log.debug(f"Normal: {normal}")

        xz_base_normal = np.cross(x_axis_base, z_unit_vector)
        yz_base_normal = np.cross(y_axis_base, z_unit_vector)
        # log.debug(f"XZ base normal: {xz_base_normal}")
        # log.debug(f"YZ base normal: {yz_base_normal}")

        tilt_vec1 = proj_on_plane(xz_base_normal, normal)
        tilt_vec2 = proj_on_plane(yz_base_normal, normal)
        # log.debug(f"Tilt vector 1: {tilt_vec1}")
        # log.debug(f"Tilt vector 2: {tilt_vec2}")

        angle_1 = 90 - math.degrees(angle_between_vectors(tilt_vec1, x_axis_base))
        angle_2 = 90 - math.degrees(angle_between_vectors(tilt_vec2, y_axis_base))
        # log.debug(f"Angle #1: {angle_1}")
        # log.debug(f"Angle #2: {angle_2}")

        return np.array([angle_1, angle_2])

    def _get_displacement_2ax(self, current_block, ref_pos, positions, orientations):
        displacements = []  # a list of displacement vectors
        for i in positions:
            if i != current_block:
                displacements.append(positions[i] - ref_pos[i])

        # convert list into array
        displacements = np.reshape(displacements, (len(displacements), 3))

        # calculate mean
        mean_displacement = np.mean(displacements, axis=0)
        log.debug(f"Mean displacement: {mean_displacement/one_millimeter}")

        # calculate base vectors
        lowest_layer = self.get_lowest_layer(positions)
        log.debug(f"Lowest layer: {lowest_layer}")
        x = []
        y = []
        for id in lowest_layer:
            orientation = Quaternion(orientations[id])
            x.append(orientation.rotate(x_unit_vector))
            y.append(orientation.rotate(y_unit_vector))
        x_axis_base = self.mean_vector(x)
        y_axis_base = self.mean_vector(y)
        log.debug(f"X base: {x_axis_base}")
        log.debug(f"Y base: {y_axis_base}")

        proj_x = orth_proj(x_axis_base, mean_displacement)
        proj_x_sign = 1 if np.sign(proj_x[0]) == np.sign(x_axis_base[0]) else -1
        proj_y = orth_proj(y_axis_base, mean_displacement)
        proj_y_sign = 1 if np.sign(proj_y[0]) == np.sign(y_axis_base[0]) else -1
        log.debug(f"X_proj: {proj_x}")
        log.debug(f"Y_proj: {proj_y}")

        displacement_x = proj_x_sign * np.linalg.norm(proj_x)
        displacement_y = proj_y_sign * np.linalg.norm(proj_y)

        return np.array([displacement_x, displacement_y]) / one_millimeter

    def _get_displacement_1ax(self, current_block, ref_pos, positions, orientations):
        return np.linalg.norm(self._get_displacement_2ax(current_block, ref_pos, positions, orientations))

    def get_last_displacement_2ax(self, current_block, positions, orientations):
        return self._get_displacement_2ax(current_block, self.last_ref_positions, positions, orientations)

    def get_last_displacement_1ax(self, current_block, positions, orientations):
        return self._get_displacement_1ax(current_block, self.last_ref_positions, positions, orientations)

    def get_abs_displacement_2ax(self, current_block, positions, orientations):
        return self._get_displacement_2ax(current_block, self.ref_positions, positions, orientations)

    def get_abs_displacement_1ax(self, current_block, positions, orientations):
        return self._get_displacement_1ax(current_block, self.ref_positions, positions, orientations)

    def _get_z_rotation(self, current_block, actual_orientations, ref_orientations):
        angles = []
        for i in actual_orientations:
            if i != current_block:
                q_ref = Quaternion(ref_orientations[i])
                x_ref = q_ref.rotate(x_unit_vector)
                q_actual = Quaternion(actual_orientations[i])
                y_actual = q_actual.rotate(y_unit_vector)
                angle = math.degrees(angle_between_vectors(x_ref, y_actual)) - 90
                angles.append(angle)
        return np.mean(angles)

    def get_total_z_rotation(self, current_block):
        return self._get_z_rotation(current_block, self.ref_orientations)

    def get_last_z_rotation(self, current_block, current_orientations):
        return self._get_z_rotation(current_block, current_orientations, self.last_ref_orientations)

    def get_block_id_from_pos(self, lvl, pos, positions, orientations, origin):
        layers = self.get_layers_state(positions, orientations, origin)
        if lvl in layers:  # if any blocks of this level exist
            if pos in layers[lvl]:
                return layers[lvl][pos]
        return None



    @staticmethod
    def generate_block(number, pos_sigma, angle_sigma, spacing):
        # TODO spacing automatic calculation

        a = (block_height_min - block_height_mean) / block_height_sigma
        b = (block_height_max - block_height_mean) / block_height_sigma
        height_distribution = truncnorm(a, b, loc=block_height_mean, scale=block_height_sigma)

        a = (block_width_min - block_width_mean) / block_width_sigma
        b = (block_width_max - block_width_mean) / block_width_sigma
        width_distribution = truncnorm(a, b, loc=block_width_mean, scale=block_width_sigma)

        a = (block_length_min - block_length_mean) / block_length_sigma
        b = (block_length_max - block_length_mean) / block_length_sigma
        length_distribution = truncnorm(a, b, loc=block_length_mean, scale=block_length_sigma)

        if number % 6 < 3:  # even level
            x = 0
            y = -block_width_mean + (number % 3) * block_width_mean
            y += (number % 3) * spacing  # add spacing between blocks
            z = number // 3 * block_height_mean + block_height_mean / 2
            angle_z = self.rg.normal(0, angle_sigma)  # add disturbance to the angle
        else:  # odd level
            x = -block_width_mean + (number % 3) * block_width_mean
            x += (number % 3) * spacing  # add spacing between blocks
            y = 0
            z = number // 3 * block_height_mean + block_height_mean / 2
            angle_z = random.gauss(90, angle_sigma)  # rotate and add disturbance

        # add disturbance to mass, position and sizes
        mass = random.gauss(block_mass_mean, block_mass_sigma)
        x = random.gauss(x, pos_sigma)
        y = random.gauss(y, pos_sigma)
        [block_size_x, block_size_y, block_size_z] = [length_distribution.rvs()/2, width_distribution.rvs()/2, height_distribution.rvs()/2]

        # WARNING: debugging code!
        # if number == 0:
        #     log.warning("The size of the first block is changed!")
        #     block_size_z = (block_height_min / 2) * 0.99

        Tower.block_sizes.append(np.array([block_size_x * 2, block_size_y * 2, block_size_z * 2]))
        s = f'''
                    <body name="block{number}" pos="{x} {y} {z}" euler="0 0 {angle_z}">
                        <freejoint name="{Tower.block_prefix + "_" + str(number) + "_joint"}"/>
                        <geom mass="{mass}" pos="0 0 0" class="block" size="{block_size_x} {block_size_y} {block_size_z}" type="box" material="mat_block{number}"/>
                    </body>'''
        return s

