from constants import *
import math
from pyquaternion import Quaternion
from tower import Tower
import time
from utils.utils import get_intermediate_rotations
import logging
import colorlog

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

class Extractor:

    size = 0.005 * scaler
    width = 0.050 * scaler
    finger_length = 0.020 * scaler
    # finger_mass = 0.0125 * scaler ** 3
    finger_mass = 100
    finger_kp = finger_mass * 10000
    finger_damping = 2 * finger_mass * math.sqrt(finger_kp / finger_mass)  # critical damping
    HOME_POS = np.array([-0.3, 0, 10 * size / scaler]) * scaler

    # base_mass = 0.5 * scaler ** 3
    base_mass = 1000
    total_mass = base_mass + 2 * finger_mass
    base_kp = total_mass * 1000
    # base_kp = 1000
    base_damping = 2 * total_mass * math.sqrt(base_kp / total_mass)  # critical damping
    # base_damping = 0

    finger1_pos = 0
    finger2_pos = 0

    distance_between_fingers_close_narrow = block_width_mean * 0.9
    distance_between_fingers_open_narrow = block_width_mean * 1.6
    distance_between_fingers_close_wide = block_length_min * 0.9
    distance_between_fingers_open_wide = block_length_mean * 1.2

    grasping_depth = block_width_mean/3*2

    translation_to_move_slowly = np.zeros(3)
    intermediate_rotations_num = 600 * 2
    intermediate_rotations = []

    counter = 0


    speed_fast = 1
    speed_slow = 1
    rotational_speed_fast = 3.14
    rotational_speed_slow = 3.14


    def __init__(self, sim, tower: Tower):
        self.sim = sim
        self.x = self.HOME_POS[0]
        self.y = self.HOME_POS[1]
        self.z = self.HOME_POS[2]
        # self.q = Quaternion(axis=[0, 0, 1], degrees=180)
        self.q = Quaternion([1, 0, 0, 0])
        self.tower = tower
        self.t = 0
        self.speed = 0.01  # in mm/timestep

    def second_order_system_step_response(self, t):
        w_n = 25
        return 1 - math.exp(-w_n * t) - w_n * t * math.exp(-w_n * t) if t > 0.001 else 0

    def set_position_vel(self, pos):
        # calculate offset
        offset = (2 * Extractor.finger_length + Extractor.size) * self.q.rotate(-x_unit_vector)

        # update position
        self.sim.data.ctrl[6] = pos[0] - self.HOME_POS[0] - offset[0]
        self.sim.data.ctrl[7] = pos[1] - self.HOME_POS[1] - offset[1]
        self.sim.data.ctrl[8] = pos[2] - self.HOME_POS[2] - offset[2]

        # update velocity
        actual_pos = self.get_position_vel()
        x_diff = pos[0] - actual_pos[0]  # pos[0] - self.HOME_POS[0] - offset[0] - actual_pos[0] + self.HOME_POS[0] + offset[0]
        y_diff = pos[1] - actual_pos[1]
        z_diff = pos[2] - actual_pos[2]
        x_diff_sign = math.copysign(1, x_diff)
        y_diff_sign = math.copysign(1, y_diff)
        z_diff_sign = math.copysign(1, z_diff)
        self.sim.data.ctrl[12] = x_diff_sign * self.second_order_system_step_response(abs(x_diff)) * self.speed_fast
        self.sim.data.ctrl[13] = y_diff_sign * self.second_order_system_step_response(abs(y_diff)) * self.speed_fast
        self.sim.data.ctrl[14] = z_diff_sign * self.second_order_system_step_response(abs(z_diff)) * self.speed_fast

    def set_position(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

    def get_position_vel(self):
        return np.array([
            self.sim.data.get_body_xpos('extractor')[0],
            self.sim.data.get_body_xpos('extractor')[1],
            self.sim.data.get_body_xpos('extractor')[2]
        ])

    def get_position(self):
        return np.array([self.x, self.y, self.z])

    def get_orientation_vel(self):
        return self.sim.data.get_body_xquat('extractor')

    def get_orientation(self):
        return self.q

    def set_orientation_vel(self, q: Quaternion):
        # update orientation
        yaw_pitch_roll = q.yaw_pitch_roll
        self.sim.data.ctrl[9] = yaw_pitch_roll[2]
        self.sim.data.ctrl[10] = yaw_pitch_roll[1]
        self.sim.data.ctrl[11] = yaw_pitch_roll[0]

        # update rotational velocities
        actual_quat = self.get_orientation_vel()
        actual_quat = Quaternion(actual_quat)
        actual_ypr = actual_quat.yaw_pitch_roll
        roll_diff = actual_ypr[2] - yaw_pitch_roll[2]
        pitch_diff = actual_ypr[1] - yaw_pitch_roll[1]
        yaw_diff = actual_ypr[0] - yaw_pitch_roll[0]
        roll_diff_sign = math.copysign(1, roll_diff)
        pitch_diff_sign = math.copysign(1, pitch_diff)
        yaw_diff_sign = math.copysign(1, yaw_diff)

        self.sim.data.ctrl[15] = roll_diff_sign * self.second_order_system_step_response(roll_diff) * self.rotational_speed_fast
        self.sim.data.ctrl[16] = pitch_diff_sign * self.second_order_system_step_response(pitch_diff) * self.rotational_speed_fast
        self.sim.data.ctrl[17] = yaw_diff_sign * self.second_order_system_step_response(yaw_diff) * self.rotational_speed_fast

    def set_orientation(self, q: Quaternion):
        self.q = q

    def update_positions(self, t):
        # update timestep
        self.t = t

        self.set_position_vel(self.get_position())
        self.set_orientation_vel(self.get_orientation())

        # translation_distance = np.linalg.norm(self.translation_to_move_slowly)
        # if translation_distance > 0.00001:
        #     translation_direction = self.translation_to_move_slowly / translation_distance
        #     if translation_distance > self.speed:
        #         self.translate(translation_direction * self.speed)
        #         self.translation_to_move_slowly -= self.speed * translation_direction
        #     else:
        #         self.translate(self.translation_to_move_slowly)
        #         self.translation_to_move_slowly = np.zeros(3)
        # else:
        #     self.translate(self.translation_to_move_slowly)
        #     self.translation_to_move_slowly = np.zeros(3)
        #
        # # rotate slowly
        # if self.intermediate_rotations:  # if the list is not empty
        #     self.q = self.intermediate_rotations[0]
        #     self.intermediate_rotations = self.intermediate_rotations[1:]  # remove first element
        #
        # # calculate offset
        # offset = (2 * Extractor.finger_length + Extractor.size) * self.q.rotate(-x_unit_vector)
        #
        # # update position
        # self.sim.data.ctrl[6] = self.x - self.HOME_POS[0] - offset[0]
        # self.sim.data.ctrl[7] = self.y - self.HOME_POS[1] - offset[1]
        # self.sim.data.ctrl[8] = self.z - self.HOME_POS[2] - offset[2]
        # # update orientation
        # yaw_pitch_roll = self.q.yaw_pitch_roll
        # self.sim.data.ctrl[9] = yaw_pitch_roll[2]
        # self.sim.data.ctrl[10] = yaw_pitch_roll[1]
        # self.sim.data.ctrl[11] = yaw_pitch_roll[0]
        #
        # # update finger positions
        # self.sim.data.ctrl[12] = self.finger1_pos
        # self.sim.data.ctrl[13] = self.finger2_pos

    def move_in_direction(self, direction, distance=one_millimeter):  # 'left', 'right', 'forward', 'backwards', 'up', 'down'

        assert direction == 'left' or direction == 'right' or direction == 'forward' or \
               direction == 'backwards' or direction == 'up' or direction == 'down', 'Unknown direction!'

        translation = None
        if direction == 'left':
            translation = -np.array(np.array(self.q.rotate(y_unit_vector)) * distance)
        if direction == 'right':
            translation = +np.array(np.array(self.q.rotate(y_unit_vector)) * distance)
        if direction == 'forward':
            translation = -np.array(self.q.rotate(x_unit_vector)) * distance
        if direction == 'backwards':
            translation = +np.array(self.q.rotate(x_unit_vector)) * distance
        if direction == 'up':
            translation = +np.array(self.q.rotate(z_unit_vector)) * distance
        if direction == 'down':
            translation = -np.array(self.q.rotate(z_unit_vector)) * distance
        self.x += translation[0]
        self.y += translation[1]
        self.z += translation[2]

    def set_finger_distance(self, distance):
        offset = distance / 2
        self.finger1_pos = -offset + (Extractor.width - 2 * Extractor.size)
        self.finger2_pos = offset - (Extractor.width - 2 * Extractor.size)

    def open_narrow(self):
        self.set_finger_distance(self.distance_between_fingers_open_narrow)

    def close_narrow(self):
        self.set_finger_distance(self.distance_between_fingers_close_narrow)

    def open_wide(self):
        self.set_finger_distance(self.distance_between_fingers_open_wide)

    def close_wide(self):
        self.set_finger_distance(self.distance_between_fingers_close_wide)

    # Returns quarter in which the extractor is located
    # The quarters are defined around the origin as follows:
    #
    # y ^                  \ 1/
    #   |                   \/
    #   |                2  /\  4
    #    ------->          / 3\
    #           x
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

    def _get_stopovers(self, start_quarter, end_quarter, height):
        assert start_quarter in [1, 2, 3, 4] and end_quarter in [1, 2, 3, 4]
        log.debug(f"Start quarter: {start_quarter}.")
        log.debug(f"End quarter: {end_quarter}.")
        distance_from_origin = block_length_mean * 2

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
        if start_quarter == 1 and end_quarter == 1:
            stopover = y_unit_vector * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 2 and end_quarter == 2:
            stopover = -x_unit_vector * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 3 and end_quarter == 3:
            stopover = -y_unit_vector * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 4 and end_quarter == 4:
            stopover = x_unit_vector * distance_from_origin
            stopovers = [np.array(stopover) + np.array([0, 0, height])]

        return stopovers

    def go_home(self):
        start_quarter = self._get_quarter(self.get_position())
        end_quarter = self._get_quarter(self.HOME_POS)
        stopovers = self._get_stopovers(start_quarter, end_quarter, self.z)
        for stopover in stopovers:
            self.set_position(stopover)
            self._sleep_simtime(0.2)
        self.set_position(self.HOME_POS)

    def put_on_top(self, placing_pose, num_of_blocks):
        heighest_block_z = self.tower.get_position(self.tower.get_highest_block_id())[2]
        height = heighest_block_z + block_height_mean*2

        # calculate target
        pos = placing_pose[:3]
        orientation = Quaternion(placing_pose[3:])
        if num_of_blocks == 3:
            # offset because block's origin is located not on extractor origin
            target_pos = pos + orientation.rotate(x_unit_vector) * (block_length_mean/2 - self.grasping_depth)
        else:
            # additional offset because of the another blocks on the top
            target_pos = pos + orientation.rotate(x_unit_vector) * (block_length_mean/2 - self.grasping_depth + self.grasping_depth*1.3)

        intermediate_target = target_pos + np.array([0, 0, block_height_max*1.5])

        # get stopovers
        stopovers = self._get_stopovers(self._get_quarter(self.get_position()),
                                        self._get_quarter(intermediate_target),
                                        height)

        for stop in stopovers:
            self.move_slowly(stop)

        self.rotate_slowly(orientation)
        self.move_slowly(intermediate_target)
        self.move_slowly(target_pos)
        self._sleep_simtime(0.2)
        self.open_narrow()

    def pull(self, block_id):
        block_quat = Quaternion(self.tower.get_orientation(block_id)) * Quaternion(axis=[0, 0, 1], degrees=180)
        block_pos = np.array(self.tower.get_position(block_id))
        block_x_face_normal_vector = np.array(block_quat.rotate(x_unit_vector))
        target = np.array(block_pos + block_length_mean * block_x_face_normal_vector)
        self.translate_slowly(target - block_pos)

    def move_from_block(self, block_id):
        block_quat = Quaternion(self.tower.get_orientation(block_id)) * Quaternion(axis=[0, 0, 1], degrees=180)
        block_pos = np.array(self.tower.get_position(block_id))
        block_x_face_normal_vector = np.array(block_quat.rotate(x_unit_vector))
        target = np.array(block_pos + (block_length_mean + (
                Extractor.finger_length * 2 + Extractor.size)) * block_x_face_normal_vector)
        self.set_position(target)

    def move_from_block_vert(self):
        self.translate_slowly(
            np.array([
                0,
                0,
                0.03
            ]) * scaler
        )

    def translate(self, v_translation):
        assert isinstance(v_translation, np.ndarray), "Translation vector must be of type np.ndarray"
        assert v_translation.size == 3, 'Incorrect translation vector dimensionality.'
        self.x += v_translation[0]
        self.y += v_translation[1]
        self.z += v_translation[2]

    def translate_slowly(self, translation):
        self.translation_to_move_slowly = translation
        self.wait_until_translation_done()

    def move_slowly(self, pos):
        self.translation_to_move_slowly = pos - self.get_position()
        self.wait_until_translation_done()

    def rotate_slowly(self, q_end):
        ypr1 = self.get_orientation().yaw_pitch_roll
        ypr2 = q_end.yaw_pitch_roll
        difference = math.degrees(abs(ypr1[0] - ypr2[0]))
        self.intermediate_rotations = get_intermediate_rotations(self.get_orientation(),
                                                                 q_end,
                                                                 int(difference*5))

        self.wait_until_rotation_done()

    def wait_until_rotation_done(self):
        while self.intermediate_rotations:  # while the list is not empty
            self._sleep_simtime(0.2)

    def wait_until_translation_done(self):
        while np.any(self.translation_to_move_slowly):
            self._sleep_simtime(0.2)

    def extract_and_put_on_top(self, id):

        # extracting
        self.move_to_block(id)
        self.close_narrow()
        self._sleep_simtime(0.2)
        self.pull(id)
        self._sleep_simtime(0.3)

        # move through stopovers (ignore the last)
        stopovers = self._get_stopovers(
            self._get_quarter(self.get_position()),
            self._get_quarter(zwischanablage_pos),
            zwischanablage_pos[2]
        )
        for i in range(len(stopovers) - 1):  # ignore the last stopover
            self.move_slowly(stopovers[i])

        # put on the zwischenablage and regrasp
        self.put_on_zwischenablage()
        self._sleep_simtime(0.5)
        self.take_from_zwischenablage()
        log.debug("Taken")

        # get placing position and orientation
        pose_info = self.tower.get_placing_pose(self.tower.get_positions())
        pos_with_tolerance = pose_info['pos_with_tolerance']
        pos = pose_info['pos']
        block_orientation = pose_info['orientation']
        stopovers = self._get_stopovers(
            self._get_quarter(self.get_position()),
            self._get_quarter(pos_with_tolerance),
            pos_with_tolerance[2] + 0.03 * scaler
        )

        # move through stopovers
        for stop in stopovers:
            self.move_slowly(stop)

        # rotate
        gripper_orientation = block_orientation * Quaternion(axis=x_unit_vector, degrees=90) * Quaternion(axis=y_unit_vector, degrees=90)
        self.rotate_slowly(gripper_orientation)

        # move to the target position
        self.move_slowly(pos)
        self._sleep_simtime(0.1)

        # palce
        self.open_wide()

        # move from block
        self.move_from_block_vert()
        self._sleep_simtime(0.3)
        self.go_home()

    def put_on_zwischenablage(self):
        # offset of the gripper relative to the zwischenablage
        offset_vector = np.array([
            zwischanablage_base_size[0] - block_width_mean/2,
            zwischanablage_base_size[1] + block_width_mean/3,
            zwischanablage_base_size[2] + block_height_mean
        ])
        rotation = Quaternion(zwischenablage_quat_elem) * Quaternion(axis=z_unit_vector, degrees=90)
        offset_vector = Quaternion(zwischenablage_quat_elem).rotate(offset_vector)
        stopover = zwischanablage_pos + np.array([0, 0, 0.1]) * scaler
        self.move_slowly(stopover)
        self.rotate_slowly(rotation)
        target = zwischanablage_pos + offset_vector
        self.move_slowly(target)
        self.open_narrow()


    def take_from_zwischenablage(self):
        # move extractor to a intermediate position to avoid collisions
        intermediate_translation = np.array([
            zwischanablage_base_size[0] - block_width_mean,
            0,
            block_length_mean
        ])
        intermediate_translation = zwischenablage_quat.rotate(intermediate_translation)
        self.move_slowly(zwischanablage_pos + intermediate_translation)

        # orient extractor to grip the block on the long side
        orientation = Quaternion(zwischenablage_quat_elem) * Quaternion(axis=y_unit_vector, degrees=-90)
        self.rotate_slowly(orientation)

        # open the gripper
        self.open_wide()

        # translate to the gripping position
        translation = np.array([
            zwischanablage_base_size[0] - block_width_mean/2,
            0,
            zwischanablage_base_size[2] + block_height_mean/2
        ])
        translation = zwischenablage_quat.rotate(translation)
        self.move_slowly(zwischanablage_pos + translation)

        # close the gripper
        self.close_wide()
        self._sleep_simtime(0.3)


        # move slightly upwards
        self.translate_slowly(np.array([0, 0, 0.05]) * scaler)

    # The path is calculated according to the quarters in which extractor and the target are located
    def move_to_block(self, num):
        block_quat = Quaternion(self.tower.get_orientation(num)) * Quaternion(axis=[0, 0, 1], degrees=180)
        block_pos = np.array(self.tower.get_position(num))
        block_x_face_normal_vector = np.array(block_quat.rotate(x_unit_vector))

        target = np.array(block_pos + (block_length_mean/2 - self.grasping_depth) * block_x_face_normal_vector)
        intermediate_target = target + (block_length_mean/2) * block_x_face_normal_vector

        stopovers = self._get_stopovers(self._get_quarter(self.get_position()),
                                        self._get_quarter(target),
                                        target[2])

        for stopover in stopovers:
            self.set_position(stopover)
            self._sleep_simtime(0.2)

        self.set_orientation(block_quat)
        self._sleep_simtime(0.3)
        log.debug('Orientation set!')
        self.set_position(intermediate_target)
        self._sleep_simtime(0.2)
        log.debug('Intermediate position set')

        self.set_position(target)
        self._sleep_simtime(0.2)
        log.debug('Position set')

    @staticmethod
    def _point_projection_on_line(line_point1, line_point2, point):
        ap = point - line_point1
        ab = line_point2 - line_point1
        result = line_point1 + np.dot(ap, ab) / np.dot(ab, ab) * ab
        return result

    def move_along_own_axis(self, axis, distance):  # distance can be negative
        assert axis in ['x', 'y', 'z'], "Wrong axis!"

        if axis == 'x':
            unit_vector = x_unit_vector
        if axis == 'y':
            unit_vector = y_unit_vector
        if axis == 'z':
            unit_vector = z_unit_vector

        translation_direction = np.array(self.q.rotate(unit_vector))
        end_point = np.array([self.x, self.y, self.z]) + translation_direction * distance
        self.set_position(pos=end_point)

    def move_along_own_axis_towards_point(self, axis, point):
        assert axis in ['x', 'y', 'z'], "Wrong axis!"

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

    def _sleep_timesteps(self, n):
        current_timestep = self.t
        while self.t < current_timestep + n:
            time.sleep(0.05)


    def _sleep_simtime(self, t):
        current_time = self.t * g_timestep
        while self.t * g_timestep < current_time + t:
            time.sleep(0.05)

    @staticmethod
    def generate_xml():
        return f'''<body name="extractor" pos="{Extractor.HOME_POS[0]} {Extractor.HOME_POS[1]} {Extractor.HOME_POS[2]}" euler="0 0 0">
                            <joint name="extractor_slide_x" type="slide" pos="0 0 0" axis="1 0 0" damping="{Extractor.base_damping}"/>
                            <joint name="extractor_slide_y" type="slide" pos="0 0 0" axis="0 1 0" damping="{Extractor.base_damping}"/>
                            <joint name="extractor_slide_z" type="slide" pos="0 0 0" axis="0 0 1" damping="{Extractor.base_damping}"/>
                            <joint name="extractor_hinge_x" type="hinge" axis="1 0 0" damping="{Extractor.base_damping}" pos ="0 0 0"/>
                            <joint name="extractor_hinge_y" type="hinge" axis="0 1 0" damping="{Extractor.base_damping}" pos ="0 0 0"/>
                            <joint name="extractor_hinge_z" type="hinge" axis="0 0 1" damping="{Extractor.base_damping}" pos ="0 0 0"/>
                            
                            <geom type="box" pos="0 0 0" size="{Extractor.size} {Extractor.width} {Extractor.size}" mass="{Extractor.base_mass}"/>
                            <body name="finger1" pos="{-Extractor.finger_length - Extractor.size} {-Extractor.width + Extractor.size} 0">
                                <joint name="finger1_joint" pos="0 0 0" type="slide" axis="0 1 0" damping="{Extractor.finger_damping}"/>
                                <geom type="box" size="{Extractor.finger_length} {Extractor.size} {Extractor.size}" mass="{Extractor.finger_mass}" friction="{1*5} {0.005*5} {0.0001*5}"/>
                            </body>
                            <body name="finger2" pos="{-Extractor.finger_length - Extractor.size} {Extractor.width - Extractor.size} 0">
                                <joint name="finger2_joint" pos="0 0 0" type="slide" axis="0 1 0" damping="{Extractor.finger_damping}"/>
                                <geom type="box" size="{Extractor.finger_length} {Extractor.size} {Extractor.size}" mass="{Extractor.finger_mass}" friction="{1*5} {0.005*5} {0.0001*5}"/>
                            </body>                            
                            <geom type="box" size="0.4 0.4 0.4" mass="0.1" rgba="1 0 0 1" pos="0 0 0"/>
                        </body>'''
