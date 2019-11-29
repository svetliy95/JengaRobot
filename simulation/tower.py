import numpy as np
from constants import *
from pyquaternion import Quaternion
from numpy.random import normal
from scipy.stats import truncnorm

class Tower:
    pos = np.array([0, 0, 0])
    block_num = 54
    block_prefix = "block"
    sim = None
    starting_positions = []
    block_sizes = []

    def __init__(self, sim):
        self.sim = sim
        pass

    def get_position(self, num):
        return self.sim.data.get_body_xpos(self.block_prefix + str(num))

    def get_orientation(self, num):
        assert num < self.block_num
        return self.sim.data.get_body_xquat(self.block_prefix + str(num))

    def get_highest_block_num(self):
        max_z = 0
        max_block_num = -1
        for i in range(self.block_num):
            if self.get_position(i)[2] > max_z:
                max_block_num = i

        return max_block_num

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
        num = self.get_highest_block_num()
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
        num = self.get_highest_block_num()
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
        print([block_size_x, block_size_y, block_size_z])

        Tower.starting_positions.append(np.array([x, y, z + block_height_mean / 2]))
        Tower.block_sizes.append(np.array([block_size_x * 2, block_size_y * 2, block_size_z * 2]))
        # print(block_size_z * 2 / one_millimeter)

        s = f'''
                    <body name="block{number}" pos="{x} {y} {z + block_height_mean / 2}" euler="0 0 {angle_z}">
                        <joint type="slide" axis="1 0 0" pos ="0 0 0"/>
                        <joint type="slide" axis="0 1 0" pos ="0 0 0"/>
                        <joint type="slide" axis="0 0 1" pos ="0 0 0"/>
                        <joint type="hinge" axis="1 0 0"  pos ="0 0 0"/>
                        <joint type="hinge" axis="0 1 0" pos ="0 0 0"/>
                        <joint type="hinge" axis="0 0 1"  pos ="0 0 0"/>
                        <geom mass="{mass}" pos="0 0 0" class="block" size="{block_size_x} {block_size_y} {block_size_z}" type="box"/>
                    </body>'''
        return s
