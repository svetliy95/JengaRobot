import numpy as np
from constants import *
from pyquaternion import Quaternion
from numpy.random import normal

class Tower:
    pos = np.array([0, 0, 0])
    block_num = 54
    block_prefix = "block"
    sim = None
    starting_positions = []

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
        [block_size_x, block_size_y, block_size_z] = normal(
            [block_length_mean / 2, block_width_mean / 2, block_height_mean / 2],
            [block_length_sigma, block_width_sigma, block_height_sigma])

        Tower.starting_positions.append(np.array([x, y, z + block_height_mean / 2]))

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

