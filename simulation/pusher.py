from pyquaternion import Quaternion
import numpy as np
from math import sqrt
import time
from constants import *
from threading import Thread

class Pusher():
    # pusher start pos
    START_X = 0.15 * scaler
    START_Y = 0 * scaler
    START_Z = 0.005 * scaler

    # pusher parameter
    pusher_size = 0.005 * scaler
    pusher_mass = 0.0125 * scaler ** 3
    pusher_base_mass = 0.125 * scaler ** 3
    pusher_kp = pusher_mass * 1000
    pusher_base_kp = pusher_base_mass * 1000
    pusher_base_damping = 2 * pusher_base_mass * sqrt(pusher_base_kp / pusher_base_mass)  # critical damping
    pusher_damping = 2 * pusher_mass * sqrt(pusher_kp / pusher_mass)  # critical damping




    def __init__(self, sim):
        self.sim = sim
        self.blocks_num = g_blocks_num
        self.x = 0
        self.y = 0
        self.z = 0
        self.q = Quaternion(1, 0, 0, 0)
        self.current_block = 0

        # pusher speed
        self.speed = 0.01  # in mm/timestep
        self.translation_to_push = np.zeros(3)  # in mm
        self.t = 0

    def update_position(self, t):
        self.t = t
        # move if necessary
        # print(self.translation_to_push)
        translation_distance = np.linalg.norm(self.translation_to_push)
        if translation_distance > 0.00001:
            translation_direction = self.translation_to_push / translation_distance
            if translation_distance > self.speed:
                self.translate(translation_direction * self.speed)
                self.translation_to_push -= self.speed * translation_direction
            else:
                self.translate(self.translation_to_push)
                self.translation_to_push = np.zeros(3)

        yaw_pitch_roll = self.q.yaw_pitch_roll
        self.sim.data.ctrl[0] = self.x
        self.sim.data.ctrl[1] = self.y
        self.sim.data.ctrl[2] = self.z
        self.sim.data.ctrl[3] = yaw_pitch_roll[2]
        self.sim.data.ctrl[4] = yaw_pitch_roll[1]
        self.sim.data.ctrl[5] = yaw_pitch_roll[0]

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_position(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

    def set_orientation(self, q):
        self.q = q

    def move_pusher_to_block(self, block_num):
        gap = 5 * one_millimeter

        # get orientation of the target block as a quaternion
        block_quat = Quaternion(self.sim.data.sensordata[3 * self.blocks_num + block_num * 4 + 0],
                                self.sim.data.sensordata[3 * self.blocks_num + block_num * 4 + 1],
                                self.sim.data.sensordata[3 * self.blocks_num + block_num * 4 + 2],
                                self.sim.data.sensordata[3 * self.blocks_num + block_num * 4 + 3])
        block_x_face_normal_vector = block_quat.rotate(x_unit_vector)
        target_x = self.sim.data.sensordata[block_num * 3 + 0] - self.START_X + (
                    block_length_mean / 2 + self.pusher_size + pusher_spring_length + gap) * block_x_face_normal_vector[0]
        target_y = self.sim.data.sensordata[block_num * 3 + 1] - self.START_Y + (
                    block_length_mean / 2 + self.pusher_size + pusher_spring_length + gap) * block_x_face_normal_vector[1]
        target_z = self.sim.data.sensordata[block_num * 3 + 2] - self.START_Z + (
                    block_length_mean / 2 + self.pusher_size + pusher_spring_length + gap) * block_x_face_normal_vector[2]

        target = np.array([target_x, target_y, target_z]) + np.array(block_x_face_normal_vector) * block_length_mean

        # move pusher backwards to avoid collisions
        self.move_along_own_axis('x', block_length_mean)
        self._sleep_timesteps(30)

        # move pusher along its own y axis first to avoid collisions
        self.move_along_own_axis_towards_point('y', target)
        self._sleep_timesteps(30)

        # rotate pusher towards the block
        self.q = block_quat
        # self._sleep_timesteps(30)

        # move pusher along its own z axis first to avoid collisions
        self.move_along_own_axis_towards_point('z', target)
        # self._sleep_timesteps(30)

        # move pusher along its own y axis first to avoid collisions (needed by side changes)
        self.move_along_own_axis_towards_point('y', target)
        self._sleep_timesteps(30)

        # move towards real target
        target = np.array([target_x, target_y, target_z])
        self.set_position(target)

    def move_pusher_to_next_block(self):
        if self.current_block < g_blocks_num - 1:
            self.current_block += 1
            self.move_pusher_to_block(self.current_block)

    def move_pusher_to_previous_block(self):
        if self.current_block > 0:
            self.current_block -= 1
            self.move_pusher_to_block(self.current_block)

    @staticmethod
    def _point_projection_on_line(line_point1, line_point2, point):
        ap = point - line_point1
        ab = line_point2 - line_point1
        result = line_point1 + np.dot(ap, ab) / np.dot(ab, ab) * ab
        return result

    def translate(self, v_translation):
        assert isinstance(v_translation, np.ndarray), "Translation vector must be of type np.ndarray"
        assert v_translation.size == 3, 'Incorrect translation vector dimensionality.'
        self.x += v_translation[0]
        self.y += v_translation[1]
        self.z += v_translation[2]

    def move_pusher_in_direction(self, direction, distance=one_millimeter):  # 'left', 'right', 'forward', 'backwards', 'up', 'down'

        assert direction == 'left' or direction == 'right' or direction == 'forward' or \
               direction == 'backwards' or direction == 'up' or direction == 'down', 'Unknown direction!'

        if (direction == 'left'):
            self.translation_to_push -= np.array(np.array(self.q.rotate(y_unit_vector)) * distance)
            print(self.translation_to_push)
        if (direction == 'right'):
            self.translation_to_push += np.array(np.array(self.q.rotate(y_unit_vector)) * distance)
        if (direction == 'forward'):
            self.translation_to_push -= np.array(self.q.rotate(x_unit_vector)) * distance
        if (direction == 'backwards'):
            self.translation_to_push += np.array(self.q.rotate(x_unit_vector)) * distance
        if (direction == 'up'):
            self.translation_to_push += np.array(self.q.rotate(z_unit_vector)) * distance
        if (direction == 'down'):
            self.translation_to_push -= np.array(self.q.rotate(z_unit_vector)) * distance

    def push_forward(self, distance=45*one_millimeter):
        self.translation_to_push -= np.array([1, 0, 0]) * distance

    def move_along_own_axis_towards_point(self, axis, point):
        assert axis == 'x' or axis == 'y' or axis == 'z', "Wrong axis!"

        if axis == 'x':
            unit_vector = x_unit_vector
        if axis == 'y':
            unit_vector = y_unit_vector
        if axis == 'z':
            unit_vector = z_unit_vector

        translation_direction = self.q.rotate(unit_vector)
        projection = self._point_projection_on_line(np.array([self.x, self.y, self.z]),
                                                  np.array([self.x, self.y, self.z]) + translation_direction,
                                                  np.array([point[0], point[1], point[2]]))
        self.set_position(projection)

    def move_along_own_axis(self, axis, distance):  # distance can be negative
        assert axis == 'x' or axis == 'y' or axis == 'z', "Wrong axis!"

        if axis == 'x':
            unit_vector = x_unit_vector
        if axis == 'y':
            unit_vector = y_unit_vector
        if axis == 'z':
            unit_vector = z_unit_vector

        translation_direction = np.array(self.q.rotate(unit_vector))
        end_point = np.array([self.x, self.y, self.z]) + translation_direction * distance
        self.set_position(end_point)

    def push(self):
        # save block position
        block_pos_before = np.array([self.sim.data.sensordata[self.current_block * 3 + i] for i in range(3)])
        self.move_pusher_in_direction('forward', one_millimeter)
        self._sleep_timesteps(75)
        block_pos_after = np.array([self.sim.data.sensordata[self.current_block * 3 + i] for i in range(3)])
        current_sensor_value = -self.sim.data.sensordata[g_blocks_num * 3 + g_blocks_num * 4]

        displacement = np.linalg.norm(block_pos_after - block_pos_before) / one_millimeter

        return current_sensor_value, displacement


    def _sleep_timesteps(self, n):
        current_timestep = self.t
        while self.t < current_timestep + n:
            time.sleep(0.05)


    def _sleep_simtime(self, t):
        current_time = self.t * g_timestep
        while self.t * g_timestep < current_time + t:
            time.sleep(0.05)
