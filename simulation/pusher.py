from pyquaternion import Quaternion
import numpy as np
from math import sqrt
import time
from constants import *
from utils.utils import *
from threading import Thread
import logging
import colorlog
from tower import Tower
import sys
import traceback
import copy

# specify logger
# DEBUG: Detailed information, typically of interest only when diagnosing problems.
# INFO: Confirmation that things are working as expected.
# WARNING: An indication that something unexpected happened, or indicative of some problem in
# the near future (e.g. ‘disk space low’). The software is still working as expected.
# ERROR: Due to a more serious problem, the software has not been able to perform some function.
# CRITICAL: A serious error, indicating that the program itself may be unable to continue running.

log = logging.Logger(__name__)
# formatter = colorlog.ColoredFormatter('%(log_color)s%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')
formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(funcName)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
log.addHandler(stream_handler)

log = logging.Logger("file_logger")
file_formatter = logging.Formatter('%(levelname)sMain process:PID:%(process)d:%(funcName)s:%(message)s')
file_handler = logging.FileHandler(filename='pusher.log', mode='w')
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)

class Pusher():
    # pusher start pos
    START_X = 0.15 * scaler
    START_Y = 0 * scaler
    START_Z = 0.01 * scaler

    # pusher parameter
    pusher_size = 0.005 * scaler
    pusher_mass = 0.0125 * scaler ** 3
    pusher_base_mass = 0.125 * scaler ** 3
    pusher_kp = pusher_mass * 1000
    pusher_base_kp = pusher_base_mass * 10000
    pusher_base_damping = 4 * (pusher_base_mass + pusher_mass) * sqrt(pusher_base_kp / (pusher_base_mass + pusher_mass))  # critical damping
    pusher_damping = 2 * pusher_mass * sqrt(pusher_kp / pusher_mass)  # critical damping

    translation_tolerance = 10 * one_millimeter
    rotation_tolerance = 5  # in degrees

    is_welded = False




    def __init__(self, sim, tower: Tower, env):
        self.sim = sim
        self.tower = tower
        self.blocks_num = g_blocks_num
        self.x = self.START_X
        self.y = self.START_Y
        self.z = self.START_Z
        self.q = Quaternion(1, 0, 0, 0)
        self.current_block = 0

        # pusher speed
        self.speed = 0.01  # in mm/timestep
        self.translation_to_push = np.zeros(3)  # in mm
        self.t = 0

        self.jenga_env = env

    # returns the actual position extracted from the engine
    # offset is used to set the origin between fingers ends
    def get_actual_pos(self):
        actual_orientation = self.get_actual_orientation()
        offset = -(pusher_spring_length + Pusher.pusher_size) * actual_orientation.rotate(x_unit_vector)
        return self.sim.data.get_body_xpos('pusher_base') + offset

    def get_actual_orientation(self):
        return Quaternion(self.sim.data.get_body_xquat('pusher_base'))

    # returns the position set by user
    def get_position(self):
        return np.array([self.x, self.y, self.z])

    def get_orientation(self):
        return self.q

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
        actual_orientation = self.get_actual_orientation()
        offset = -(pusher_spring_length + Pusher.pusher_size) * actual_orientation.rotate(x_unit_vector)
        self.sim.data.ctrl[0] = self.x - self.START_X - offset[0]
        self.sim.data.ctrl[1] = self.y - self.START_Y - offset[1]
        self.sim.data.ctrl[2] = self.z - self.START_Z - offset[2]
        self.sim.data.ctrl[3] = yaw_pitch_roll[2]
        self.sim.data.ctrl[4] = yaw_pitch_roll[1]
        self.sim.data.ctrl[5] = yaw_pitch_roll[0]

    def set_position(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.wait_until_translation_done()

    def set_orientation(self, q):
        self.q = q
        self.wait_until_rotation_done()

    def _get_stopovers(self, start_quarter, end_quarter, height):
        assert start_quarter in [1, 2, 3, 4] and end_quarter in [1, 2, 3, 4]
        distance_from_origin = block_length_mean * 3

        stopovers = []
        if start_quarter == 1 and end_quarter == 2:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(y_unit_vector) * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 1 and end_quarter == 4:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(x_unit_vector) * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 2 and end_quarter == 1:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(y_unit_vector) * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 2 and end_quarter == 3:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-x_unit_vector) * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 3 and end_quarter == 2:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-x_unit_vector) * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 3 and end_quarter == 4:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-y_unit_vector) * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 4 and end_quarter == 3:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-y_unit_vector) * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 4 and end_quarter == 1:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(x_unit_vector) * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 2 and end_quarter == 4:
            stopovers = self._get_stopovers(2, 3, height) + self._get_stopovers(3, 4, height)
        if start_quarter == 4 and end_quarter == 2:
            stopovers = self._get_stopovers(4, 1, height) + self._get_stopovers(1, 2, height)
        if start_quarter == 1 and end_quarter == 3:
            stopovers = self._get_stopovers(1, 2, height) + self._get_stopovers(2, 3, height)
        if start_quarter == 3 and end_quarter == 1:
            stopovers = self._get_stopovers(3, 4, height) + self._get_stopovers(4, 1, height)
        if start_quarter == end_quarter:
            stopovers = []

        return stopovers

    @staticmethod
    def _get_quarter(pos):
        assert pos.size == 3, "Argument 'pos' is not a 3d array!"
        x = pos[0]
        y = pos[1]
        if y >= abs(x):
            return 1
        if y <= -abs(x):
            return 3
        if x >= 0 and abs(y) < abs(x):
            return 4
        if x < 0 and abs(y) < abs(x):
            return 2

    def move_to_block(self, block_num):
        log.debug(f"Move to block #1")
        # set current block
        self.current_block = block_num

        # set gap between the pusher and the block
        gap = 3 * one_millimeter

        # reset last reference positions and orientations
        self.tower.last_ref_positions = copy.deepcopy(self.tower.get_positions())
        self.tower.last_ref_orientations = copy.deepcopy(self.tower.get_orientations())

        log.debug(f"After deep copies")

        # get orientation of the target block as a quaternion
        block_quat = Quaternion(self.tower.get_orientation(block_num))
        block_pos = np.array(self.tower.get_position(block_num))

        log.debug(f"After get poses and orientations")

        # calculate pusher orientation
        first_block_end = block_pos + block_quat.rotate(x_unit_vector) * block_length_mean / 2
        second_block_end = block_pos + block_quat.rotate(-x_unit_vector) * block_length_mean / 2
        first_distance = np.linalg.norm(coordinate_axes_pos - first_block_end)
        second_distance = np.linalg.norm(coordinate_axes_pos - second_block_end)
        if first_distance > second_distance:
            offset_direction_quat = block_quat
        else:
            offset_direction_quat = block_quat * Quaternion(axis=[0, 0, 1], degrees=180)

        log.debug(f"After transformations")

        block_x_face_normal_vector = offset_direction_quat.rotate(x_unit_vector)
        target_x = block_pos[0] + (block_length_mean / 2 + gap) * block_x_face_normal_vector[0]
        target_y = block_pos[1] + (block_length_mean / 2 + gap) * block_x_face_normal_vector[1]
        target_z = block_pos[2] + (block_length_mean / 2 + gap) * block_x_face_normal_vector[2]

        target = np.array([target_x, target_y, target_z]) + np.array(block_x_face_normal_vector) * block_length_mean

        log.debug(f"After target definition")

        # move pusher backwards to avoid collisions
        log.debug(f"Before move along own axis")
        self.move_along_own_axis('x', block_length_mean)
        log.debug(f"After move along own axis")

        # move through stopovers
        stopovers = self._get_stopovers(self._get_quarter(self.get_position()),
                                        self._get_quarter(target),
                                        target[2])

        log.debug(f"After get stopovers")
        for stopover in stopovers:
            log.debug(f"Stopover")
            self.set_position(stopover)
            self._sleep_simtime(0.2)

        log.debug(f"After move to stopovers")

        # rotate pusher towards the block
        self.set_orientation(offset_direction_quat)
        log.debug(f"After set orientation")

        # move to intermediate target
        self.set_position(target)
        log.debug(f"After set position")

        # move to the end target
        target = np.array([target_x, target_y, target_z])
        self.set_position(target)
        log.debug(f"After set position #2")
        self._sleep_simtime(0.3)

        log.debug(f"Move to block #2")

    def move_pusher_to_next_block(self):
        if self.current_block < g_blocks_num - 1:
            self.current_block += 1
            self.move_to_block(self.current_block)

    def move_pusher_to_previous_block(self):
        if self.current_block > 0:
            self.current_block -= 1
            self.move_to_block(self.current_block)

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
        projection = point_projection_on_line(np.array([self.x, self.y, self.z]),
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

    def get_force(self):
        return -self.sim.data.sensordata[g_blocks_num * 3 + g_blocks_num * 4]

    def push(self):
        # save block position
        block_pos_before = self.tower.get_position(self.current_block)
        self.move_pusher_in_direction('forward', one_millimeter)
        log.warning(f"Calculate the sleeping time right!")
        self._sleep_simtime(0.225)
        current_sensor_value = self.get_force()
        block_pos_after = self.tower.get_position(self.current_block)

        displacement = np.linalg.norm(block_pos_after - block_pos_before) / one_millimeter

        return current_sensor_value, displacement

    def _sleep_timesteps(self, n):
        current_timestep = self.t
        while self.t < current_timestep + n and self.jenga_env.simulation_running():
            time.sleep(0.05)


    def _sleep_simtime(self, t):
        current_time = self.t * g_timestep
        while self.t * g_timestep < current_time + t and self.jenga_env.simulation_running():
            time.sleep(0.01)

    def wait_until_translation_done(self):
        while np.linalg.norm(self.get_actual_pos() - self.get_position()) > self.translation_tolerance and \
                self.jenga_env.simulation_running():
            self._sleep_simtime(0.1)

    def wait_until_rotation_done(self):
        while math.degrees(get_angle_between_quaternions(self.get_actual_orientation(),
                                                         self.get_orientation())) > self.rotation_tolerance and \
                self.jenga_env.simulation_running():
            self._sleep_simtime(0.1)
