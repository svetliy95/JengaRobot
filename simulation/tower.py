import numpy as np
from constants import *
from pyquaternion import Quaternion
from numpy.random import normal
from scipy.stats import truncnorm
from cv.block_localization import get_block_positions, get_camera_params
import cv2
import math
from utils.utils import *
import time
from PIL import Image


class Tower:
    pos = np.array([0, 0, 0])
    block_num = 54
    block_prefix = "block"
    sim = None
    starting_positions = []
    block_sizes = []
    placing_vert_spacing = block_height_max / 3
    placing_horiz_spacing = block_width_max * 0.1

    def __init__(self, sim, viewer):
        self.sim = sim
        self.viewer = viewer

    def get_position(self, num):
        assert num < self.block_num, "Block num is to high"
        return self.sim.data.get_body_xpos(self.block_prefix + str(num))

    def get_orientation(self, num):
        assert num < self.block_num, "Block num is to high"
        return self.sim.data.get_body_xquat(self.block_prefix + str(num))

    def _take_picture(self, cam_id):
        w = 4096
        h = 2160
        self.viewer.render(w, h, cam_id)
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

    def get_poses_cv(self, ids, return_images=True):
        assert set(ids).issubset(set(range(g_blocks_num))), "Wrong block ids!"
        im1 = self._take_picture(0)
        im2 = self._take_picture(1)
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
        height, width = im1_gray.shape

        poses, dimage1, dimage2 = get_block_positions(im1=im1_gray,
                                        im2=im2_gray,
                                        block_ids=ids,
                                        target_tag_size=block_height_mean * 0.8,  # 0.8 because the actual tag length is 8/10 of the whole length
                                        ref_tag_size=coordinate_frame_tag_size[0] * 0.8,  # 0.8 because the actual tag length is 8/10 of the whole length
                                        ref_tag_pos=coordinate_frame_tag_pos,
                                        ref_tag_id=coordinate_frame_tag_id,
                                        block_sizes=np.array([block_length_mean, block_width_mean, block_height_mean]),
                                        camera_params=get_camera_params(height, width, fovy),
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

    def get_position_cv(self, num):
        assert num < self.block_num, "Block num is to high"

        position = self.get_poses_cv([num])

        if len(position) > 0:
            return self.get_poses_cv([num])[0]
        else:
            return None

    def get_highest_block_id(self):
        max_z = 0
        max_block_num = -1
        for i in range(self.block_num):
            height = self.get_position(i)[2]
            if height > max_z:
                max_z = height
                max_block_num = i

        return max_block_num

    def get_blocks_from_highest_level(self):
        highest_block = self.get_highest_block_id()
        highest_block_pos = self.get_position(highest_block)

    def get_positions(self):
        positions = {}
        for i in range(self.block_num):
            positions[i] = self.get_position(i)

        return positions

    def get_adjacent_blocks(self, id, positions):
        root_pos = self.get_position(id)

        # search for blocks with the same height within a certain threshold
        adjacent_blocks = []
        for i in range(g_blocks_num):
            if abs(positions[i][2] - root_pos[2]) < same_height_threshold:
                adjacent_blocks.append(i)

        print(f"Adjacent blocks: {adjacent_blocks}")

        return adjacent_blocks

    def get_layers(self, positions):
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
                # blocks_to_test = [x for x in blocks_to_test if x not in layer]
                blocks_to_test = (set(blocks_to_test) - set(layer))

        return layers

    def get_full_layers(self, positions):
        layers = self.get_layers(positions)

        full_layers = [layer for layer in layers if len(layer) == 3]

        return full_layers

    def get_center_xy(self, positions):
        full_layers = self.get_full_layers(positions)
        x = []
        y = []
        for layer in full_layers:
            for id in layer:
                x.append(positions[id][0])
                y.append(positions[id][1])
        return np.array([np.mean(x), np.mean(y)])

    def get_highest_layer(self, positions):
        return self.get_adjacent_blocks(self.get_highest_block_id(), positions)

    def get_highest_full_layer(self, positions):
        max_z = -np.Inf
        highest_layer = None
        full_layers = self.get_full_layers(positions)
        for layer in full_layers:
            z = self.get_position(layer[0])[2]
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

        print(f"Possible positions: {possible_positions}")
        print(f"Distances: {distances}")
        for i in layer:
            print(f"Position of block #{i}: {positions[i]}")

        occupied_positions = {}
        for block in layer:
            occupied_positions[np.argmin(distances[block])] = block

        return occupied_positions

    def get_placing_pose(self, positions):
        # There are 3 cases:
        # 1. 3 blocks in the highest layer
        # 2. 2 blocks in the highest layer
        # 3. 1 block in the highest layer
        # And three possible positions for blocks
        # Position 1: offset in the direction of the origin
        # Position 2: central position
        # Position 3: offset in the direction away from the origin



        # calculate highest full layer mean height
        highest_full_layer = self.get_highest_full_layer(positions)
        z = []
        for i in range(3):
            z.append(self.get_position(highest_full_layer[i])[2])
        highest_full_layer_height = np.mean(z)

        # calculate mean orientation of the blocks on the highest full layer
        orientations = []
        for block_id in highest_full_layer:
            orientations.append(self.get_orientation(block_id))
        orientations = np.array(orientations)
        mean_orientation = Quaternion(np.mean(orientations, axis=0))  # get the mean orientation of the top blocks

        print(f"Orientations: {orientations}")
        print(f"Mean orientation: {mean_orientation}")
        print(f"Mean orientation ypr: {mean_orientation.yaw_pitch_roll}")

        # calculate center of the tower
        tower_center = self.get_center_xy(positions)
        tower_center_with_height = np.concatenate(
            (tower_center, np.array([highest_full_layer_height + block_height_max/2])))

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
        highest_layer = self.get_highest_layer(positions)

        # get occupied positions
        occupied_positions = self._assign_pos(highest_layer, possible_positions, positions)

        # case 1: place the block on the side near the origin perpendicular to the last 3 blocks
        if len(highest_layer) == 3:
            # block orientation must be perpendicular to the top blocks
            block_orientation = mean_orientation * Quaternion(axis=[0, 0, 1], degrees=90)

            # calculate pos
            temp_vector = mean_orientation.rotate(x_unit_vector)
            offset_direction = get_direction_towards_origin_along_vector(vec=temp_vector,
                                                                         p=tower_center_with_height,
                                                                         origin=np.array([coordinate_axes_pos_x,
                                                                                          coordinate_axes_pos_y,
                                                                                          coordinate_axes_pos_z]))

            placing_pos = tower_center_with_height + block_width_max * offset_direction + np.array([0, 0, block_height_max / 2 + Tower.placing_vert_spacing])
            placing_pos_with_tolerance = tower_center_with_height + (block_width_max + Tower.placing_horiz_spacing) * offset_direction + np.array([0, 0, block_height_max / 2 + Tower.placing_vert_spacing])

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

                placing_pos_with_tolerance = central_block_position - (block_width_max + Tower.placing_horiz_spacing) * offset_direction

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
                placing_pos = central_block_position + block_width_max * offset_direction
                placing_pos_with_tolerance = central_block_position + (
                            block_width_max + Tower.placing_horiz_spacing) * offset_direction

                block_orientation = central_block_orientation

            assert 0 in occupied_positions and 2 in occupied_positions, "Stupid blocks placing!"

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

                placing_pos = block0_pos - block_width_max * offset_direction
                placing_pos_with_tolerance = block0_pos - (block_width_max + Tower.placing_horiz_spacing) * offset_direction

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

                placing_pos = block1_pos + block_width_max * offset_direction
                placing_pos_with_tolerance = block1_pos + (block_width_max + Tower.placing_horiz_spacing) * offset_direction

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

                placing_pos = block2_pos + block_width_max * offset_direction
                placing_pos_with_tolerance = block2_pos + (block_width_max + Tower.placing_horiz_spacing) * offset_direction

                block_orientation = block2_orientation

        num_of_blocks = len(highest_layer)

        print(f"Occupied positions: {occupied_positions}")
        print(f"Highest layer: {highest_layer}")
        print(f"Highest full layer: {highest_full_layer}")

        return {'pos': placing_pos,
                'pos_with_tolerance': placing_pos_with_tolerance,
                'orientation': block_orientation,
                'num_of_blocks': num_of_blocks}

    def get_angle_to_ground(self, num):
        q = Quaternion(self.get_orientation(num))
        block_z_normal = q.rotate(z_unit_vector)

        return np.rad2deg(np.arccos(np.dot(z_unit_vector, block_z_normal)/(np.linalg.norm(block_z_normal))))

    def get_mean_angle_to_ground(self):
        angles = []
        for i in range(self.block_num):
            angles.append(self.get_angle_to_ground(i))

        return np.mean(angles)

    def get_angle_of_highest_block_to_ground(self):
        num = self.get_highest_block_id()
        return self.get_angle_to_ground(num)

    def get_displacement(self, num):
        starting_pos = self.starting_positions[num]
        return np.linalg.norm(self.get_position(num) - starting_pos)

    def get_average_displacement(self):
        displacements = []
        for i in range(self.block_num):
            displacements.append(self.get_displacement(i))

        return np.mean(displacements)

    def get_displacement_of_highest_block(self):
        num = self.get_highest_block_id()
        return self.get_displacement(num)

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
            z = number // 3 * block_height_mean
            angle_z = normal(0, angle_sigma)  # add disturbance to the angle
        else:  # odd level
            x = -block_width_mean + (number % 3) * block_width_mean
            x += (number % 3) * spacing  # add spacing between blocks
            y = 0
            z = number // 3 * block_height_mean
            angle_z = normal(90, angle_sigma)  # rotate and add disturbance

        # add disturbance to mass, position and sizes
        mass = normal(block_mass_mean, block_mass_sigma)
        [x, y] = normal([x, y], [pos_sigma, pos_sigma])
        [block_size_x, block_size_y, block_size_z] = [length_distribution.rvs()/2, width_distribution.rvs()/2, height_distribution.rvs()/2]

        Tower.starting_positions.append(np.array([x, y, z + block_height_mean / 2]))
        Tower.block_sizes.append(np.array([block_size_x * 2, block_size_y * 2, block_size_z * 2]))
        print(f"Block # {number}: " + str(Tower.block_sizes[-1]))

        s = f'''
                    <body name="block{number}" pos="{x} {y} {z + block_height_mean / 2}" euler="0 0 {angle_z}">
                        <joint type="slide" axis="1 0 0" pos ="0 0 0"/>
                        <joint type="slide" axis="0 1 0" pos ="0 0 0"/>
                        <joint type="slide" axis="0 0 1" pos ="0 0 0"/>
                        <joint type="hinge" axis="1 0 0"  pos ="0 0 0"/>
                        <joint type="hinge" axis="0 1 0" pos ="0 0 0"/>
                        <joint type="hinge" axis="0 0 1"  pos ="0 0 0"/>
                        <geom mass="{mass}" pos="0 0 0" class="block" size="{block_size_x} {block_size_y} {block_size_z}" type="box" material="mat_block{number}"/>
                    </body>'''
        return s

