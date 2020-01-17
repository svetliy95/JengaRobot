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
        return Quaternion.sym_distance(q1, q2)

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

    def get_placing_pose(self, positions):
        # there are 3 cases:
        # 1. 3 blocks in the highest layer
        # 2. 2 blocks in the highest layer
        # 3. 1 block in the highest layer

        highest_layer = self.get_highest_layer(positions)
        # calculate highest layer mean height
        z = []
        for i in range(len(highest_layer)):
            z.append(self.get_position(highest_layer[i])[2])
        highest_layer_height = np.mean(z)

        # case 1: place the block on the side near the origin perpendicular to the last 3 blocks
        if len(highest_layer) == 3:
            # calculate the mean orientation of the last 3 blocks
            orientations = []
            for block_id in highest_layer:
                orientations.append(self.get_orientation(block_id))
            mean_orientation = Quaternion(np.mean(orientations))
            block_orientation = mean_orientation  # * Quaternion(axis=[0, 0, 1], degrees=90)

            # calculate pos
            tower_center = self.get_center_xy(positions)
            tower_center_with_height = np.concatenate((tower_center, np.array([highest_layer_height + block_height_max*1.1])))
            temp_vector = mean_orientation.rotate(x_unit_vector)
            origin_projected_on_line = point_projection_on_line(tower_center_with_height, tower_center_with_height + 100 * temp_vector, origin)
            # off set from the center of the tower
            offset_direction = origin_projected_on_line - tower_center_with_height
            placing_pos = tower_center_with_height + block_width_max * offset_direction

        return np.concatenate((placing_pos, block_orientation.elements))

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

