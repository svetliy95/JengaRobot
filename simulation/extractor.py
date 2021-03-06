from constants import *
import math
from pyquaternion import Quaternion
from tower import Tower
import time
from utils.utils import get_intermediate_rotations
import logging
import colorlog
from utils.utils import get_angle_between_quaternions

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
    finger_mass = 0.0125 * scaler ** 3
    finger_kp = finger_mass * 10000
    finger_damping = 2 * finger_mass * math.sqrt(finger_kp / finger_mass)  # critical damping
    HOME_POS = np.array([-0.3, 0, 0.1]) * scaler

    base_mass = 0.5 * scaler ** 3
    total_mass = base_mass + 2 * finger_mass
    base_kp = total_mass * 1000
    base_damping_slide = 2 * total_mass * math.sqrt(base_kp / total_mass)  # critical damping
    base_damping_hinge = 2 * (2 * total_mass * math.sqrt(base_kp / total_mass))

    finger1_pos = 0
    finger2_pos = 0

    distance_between_fingers_close_narrow = block_width_mean * 0.9
    distance_between_fingers_open_narrow = block_width_mean * 1.6
    distance_between_fingers_close_wide = block_length_min * 0.9
    distance_between_fingers_open_wide = block_length_mean * 1.2

    grasping_depth = block_width_mean/3*2

    translation_to_move_slowly = np.zeros(3)
    intermediate_rotations_num = 2000
    intermediate_rotations = []

    grasping_depth_wide = block_height_mean / 2

    welded_block = -1
    is_welded = False
    translation_to_welded_block = None
    quat_to_welded_block = None
    transform_quat_for_the_translation_vec = None

    translation_tolerance = 5 * one_millimeter
    rotation_tolerance = 5  # in degrees

    def __init__(self, sim, tower: Tower, env):
        self.sim = sim
        self.x = self.HOME_POS[0]
        self.y = self.HOME_POS[1]
        self.z = self.HOME_POS[2]
        self.q = Quaternion(axis=[0, 0, 1], degrees=180)
        self.tower = tower
        self.t = 0
        self.speed = 0.01  # in mm/timestep
        self.jenga_env = env

    def second_order_system_step_response(self, t):
        w_n = 25
        return 1 - math.exp(-w_n * t) - w_n * t * math.exp(-w_n * t) if t > 0.001 else 0

    def set_position(self, pos):
        log.debug(f"Set position: {pos}")
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.wait_until_translation_done()

    # returns the position set by user
    def get_position(self):
        return np.array([self.x, self.y, self.z])

    # returns the actual position extracted from the engine
    # offset is used to set the origin between fingers ends
    def get_actual_pos(self):
        actual_quat = self.get_actual_orientation()
        offset = (2 * Extractor.finger_length + Extractor.size) * actual_quat.rotate(-x_unit_vector)
        return self.sim.data.get_body_xpos('extractor') + offset

    def get_actual_orientation(self):
        return Quaternion(self.sim.data.get_body_xquat('extractor'))

    def get_orientation(self):
        return self.q

    def set_orientation(self, q: Quaternion):
        log.debug(f"Set orientation: {q}")
        self.q = q
        self.wait_until_rotation_done()

    def update_positions(self, t):
        # update timestep
        self.t = t

        translation_distance = np.linalg.norm(self.translation_to_move_slowly)
        if translation_distance > 0.00001:
            translation_direction = self.translation_to_move_slowly / translation_distance
            if translation_distance > self.speed:
                self.translate(translation_direction * self.speed)
                self.translation_to_move_slowly -= self.speed * translation_direction
            else:
                self.translate(self.translation_to_move_slowly)
                self.translation_to_move_slowly = np.zeros(3)
        else:
            self.translate(self.translation_to_move_slowly)
            self.translation_to_move_slowly = np.zeros(3)

        # rotate slowly
        if self.intermediate_rotations:  # if the list is not empty
            self.q = self.intermediate_rotations[0]
            self.intermediate_rotations = self.intermediate_rotations[1:]  # remove first element

        # calculate offset
        offset = (2 * Extractor.finger_length + Extractor.size) * self.q.rotate(-x_unit_vector)



        # update position
        self.sim.data.ctrl[6] = self.x - self.HOME_POS[0] - offset[0]
        self.sim.data.ctrl[7] = self.y - self.HOME_POS[1] - offset[1]
        self.sim.data.ctrl[8] = self.z - self.HOME_POS[2] - offset[2]

        # update orientation
        yaw_pitch_roll = self.q.yaw_pitch_roll
        self.sim.data.ctrl[9] = yaw_pitch_roll[2]
        self.sim.data.ctrl[10] = yaw_pitch_roll[1]
        self.sim.data.ctrl[11] = yaw_pitch_roll[0]


        # update finger positions
        self.sim.data.ctrl[12] = self.finger1_pos
        self.sim.data.ctrl[13] = self.finger2_pos

        # update block's position if welded
        if self.is_welded:
            # WARNING: hardcoded indexes
            start_index = 7
            actual_quat = self.get_actual_orientation()
            pos = self.get_actual_pos() + (actual_quat * self.transform_quat_for_the_translation_vec).rotate(self.translation_to_welded_block)
            quat = self.get_actual_orientation() * self.quat_to_welded_block
            l = list(pos) + list(quat)
            # set positions
            self.sim.data.qpos[start_index + self.welded_block * 7:start_index + (self.welded_block+1) * 7] = l
            # set velocities to zero
            self.sim.data.qvel[6 + self.welded_block * 6:6 + (self.welded_block+1) * 6] = [0, 0, 0, 0, 0, 0]

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
        self.set_orientation(Quaternion(axis=[0, 0, 1], degrees=180))

    def pull(self, block_id):
        block_quat = Quaternion(self.tower.get_orientation(block_id)) * Quaternion(axis=[0, 0, 1], degrees=180)
        block_pos = np.array(self.tower.get_position(block_id))

        # calculate extractor orientation
        first_block_end = block_pos + block_quat.rotate(x_unit_vector) * block_length_mean / 2
        second_block_end = block_pos + block_quat.rotate(-x_unit_vector) * block_length_mean / 2
        first_distance = np.linalg.norm(coordinate_axes_pos - first_block_end)
        second_distance = np.linalg.norm(coordinate_axes_pos - second_block_end)
        if first_distance < second_distance:
            offset_quat = block_quat
        else:
            offset_quat = block_quat * Quaternion(axis=[0, 0, 1], degrees=180)

        block_x_face_normal_vector = np.array(offset_quat.rotate(x_unit_vector))
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
        self.wait_until_slow_translation_done()

    def move_slowly(self, pos):
        self.translation_to_move_slowly = pos - self.get_position()
        self.wait_until_slow_translation_done()

    def rotate_slowly(self, q_end):
        ypr1 = self.get_orientation().yaw_pitch_roll
        ypr2 = q_end.yaw_pitch_roll
        log.debug(f"yaw_pitch_roll: {list(map(math.degrees, ypr2))}")
        difference = math.degrees(abs(ypr1[0] - ypr2[0]))
        self.intermediate_rotations = get_intermediate_rotations(self.get_orientation(),
                                                                 q_end,
                                                                 int(difference*5))

        self.wait_until_slow_rotation_done()

    def wait_until_slow_rotation_done(self):
        while self.intermediate_rotations and \
                self.jenga_env.simulation_running():  # while the list is not empty
            self._sleep_simtime(0.2)

    def wait_until_slow_translation_done(self):
        while np.any(self.translation_to_move_slowly) and \
                self.jenga_env.simulation_running():
            self._sleep_simtime(0.2)

    def wait_until_translation_done(self):
        while np.linalg.norm(self.get_actual_pos() - self.get_position()) > self.translation_tolerance and \
                self.jenga_env.simulation_running():
            self._sleep_simtime(0.1)

    def wait_until_rotation_done(self):
        while math.degrees(get_angle_between_quaternions(self.get_actual_orientation(),
                                                         self.get_orientation())) > self.rotation_tolerance and \
                self.jenga_env.simulation_running():
            self._sleep_simtime(0.1)


    def move_block_using_magic(self, block_id, pos, quat):
        start_index = 7
        l = list(pos) + list(quat)
        self.sim.data.qpos[start_index + block_id * 7:start_index + (block_id + 1) * 7] = l
        self.sim.data.qvel[6 + block_id * 6:6 + (block_id + 1) * 6] = [0, 0, 0, 0, 0, 0]

    def extract_and_put_on_top_using_magic(self, id):
        # extracting
        self.move_to_block(id)
        self.close_narrow()
        self._sleep_simtime(0.2)
        self.pull(id)
        self._sleep_simtime(0.3)

        # weld
        self.weld_block_to_extractor(id)
        self.open_wide()
        self._sleep_simtime(0.1)

        # move through stopovers (ignore the last)
        stopovers = self._get_stopovers(
            self._get_quarter(self.get_position()),
            self._get_quarter(zwischanablage_pos),
            zwischanablage_pos[2]
        )
        for i in range(len(stopovers) - 1):  # ignore the last stopover
            self.set_position(stopovers[i])

        # put on the zwischenablage and regrasp
        # release welding in the following function
        self.put_on_zwischenablage()
        self._sleep_simtime(0.5)

        #

        # self.take_from_zwischenablage(id)
        # log.debug("Taken")

        # get placing position and orientation
        pose_info = self.tower.get_placing_pose_mujoco(self.tower.get_positions(), self.tower.get_orientations(), current_block=id)
        pos_with_tolerance = pose_info['pos_with_tolerance']
        pos = pose_info['pos']
        block_orientation = pose_info['orientation']

        print(f"Pos: {pos}")

        # place block on top using magic
        self.move_block_using_magic(id, pos, block_orientation)
        self.go_home()

    def extract_and_put_on_top(self, id):
        # extracting
        self.move_to_block(id)
        self.close_narrow()
        self._sleep_simtime(0.2)
        self.pull(id)
        self._sleep_simtime(0.3)

        # weld
        self.weld_block_to_extractor(id)
        self.open_wide()
        self._sleep_simtime(0.1)

        # move through stopovers (ignore the last)
        stopovers = self._get_stopovers(
            self._get_quarter(self.get_position()),
            self._get_quarter(zwischanablage_pos),
            zwischanablage_pos[2]
        )
        for i in range(len(stopovers) - 1):  # ignore the last stopover
            self.set_position(stopovers[i])

        # put on the zwischenablage and regrasp
        # release welding in the following function
        self.put_on_zwischenablage()
        self._sleep_simtime(0.5)
        self.take_from_zwischenablage(id)
        log.debug("Taken")

        # get placing position and orientation
        pose_info = self.tower.get_placing_pose(self.tower.get_positions(), current_block=id)
        pos_with_tolerance = pose_info['pos_with_tolerance']
        pos = pose_info['pos']
        block_orientation = pose_info['orientation']
        last_stopover = pose_info['stopover']
        stopovers = self._get_stopovers(
            self._get_quarter(self.get_position()),
            self._get_quarter(last_stopover),
            last_stopover[2]
        )

        # move through stopovers
        for stop in stopovers:
            self.set_position(stop)

        # rotate
        gripper_orientation = block_orientation * Quaternion(axis=y_unit_vector, degrees=-90) * Quaternion(axis=x_unit_vector, degrees=90)
        self.set_orientation(gripper_orientation)

        # go to the last stopover near the tower
        self.set_position(last_stopover)

        # move to the target position
        self.move_slowly(pos)
        self._sleep_simtime(0.5)

        # place
        self.open_wide()

        # release welding
        self.unweld_block()
        self._sleep_simtime(0.1)

        # move from block
        self.move_from_block_vert()
        self._sleep_simtime(0.3)
        self.go_home()

        # save new reference position and orientation
        self.tower.ref_positions[id] = pos
        self.tower.ref_orientations[id] = self.tower.get_orientation(id)

    def weld_block_to_extractor(self, id):
        # calculate quaternion q_t so that q_e * q_t = q_b
        # where q_e is the quaternion of the extractor and q_b the quaternion of the block
        q_b = Quaternion(self.tower.get_orientation(id))
        q_e = self.get_actual_orientation()
        self.quat_to_welded_block = q_e.inverse * q_b

        # calculate the position fo the block relative to the extractor
        block_pos = self.tower.get_position(id)
        extractor_pos = self.get_position()
        self.translation_to_welded_block = block_pos - extractor_pos
        self.transform_quat_for_the_translation_vec = q_e.inverse

        self.welded_block = id
        self.is_welded = True

    def unweld_block(self):
        self.is_welded = False

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
        self.set_position(stopover)
        self.set_orientation(rotation)
        target = zwischanablage_pos + offset_vector
        self.set_position(target)
        self.unweld_block()
        # self.open_narrow()


    def take_from_zwischenablage(self, id):
        # move extractor to a intermediate position to avoid collisions
        intermediate_translation = np.array([
            zwischanablage_base_size[0] - block_width_mean,
            0,
            block_length_mean
        ])
        intermediate_translation = zwischenablage_quat.rotate(intermediate_translation)
        self.set_position(zwischanablage_pos + intermediate_translation)

        # orient extractor to grip the block on the long side
        orientation = Quaternion(zwischenablage_quat_elem) * Quaternion(axis=y_unit_vector, degrees=-90)
        self.set_orientation(orientation)
        self._sleep_simtime(0.6)

        # open the gripper
        self.open_wide()

        # translate to the gripping position
        translation = np.array([
            zwischanablage_base_size[0] - block_width_mean/2,
            0,
            zwischanablage_base_size[2] + block_height_max - Extractor.grasping_depth_wide
        ])
        translation = zwischenablage_quat.rotate(translation)
        self.set_position(zwischanablage_pos + translation)

        # weld the block to the extractor
        self.weld_block_to_extractor(id)


        # close the gripper
        self.close_wide()
        self._sleep_simtime(0.3)


        # move slightly upwards
        self.set_position(self.get_position() + np.array([0, 0, 0.05]) * scaler)

    def put_on_top(self):
        raise NotImplementedError

    # The path is calculated according to the quarters in which extractor and the target are located
    def move_to_block(self, num):
        block_quat = Quaternion(self.tower.get_orientation(num)) * Quaternion(axis=[0, 0, 1], degrees=180)
        block_pos = np.array(self.tower.get_position(num))

        # calculate extractor orientation
        first_block_end = block_pos + block_quat.rotate(x_unit_vector) * block_length_mean / 2
        second_block_end = block_pos + block_quat.rotate(-x_unit_vector) * block_length_mean / 2
        first_distance = np.linalg.norm(coordinate_axes_pos - first_block_end)
        second_distance = np.linalg.norm(coordinate_axes_pos - second_block_end)
        if first_distance < second_distance:
            offset_direction_quat = block_quat
        else:
            offset_direction_quat = block_quat * Quaternion(axis=[0, 0, 1], degrees=180)

        block_x_face_normal_vector = np.array(offset_direction_quat.rotate(x_unit_vector))

        target = np.array(block_pos + (block_length_mean/2 - self.grasping_depth) * block_x_face_normal_vector)
        intermediate_target = target + (block_length_mean/2) * block_x_face_normal_vector

        stopovers = self._get_stopovers(self._get_quarter(self.get_position()),
                                        self._get_quarter(target),
                                        target[2])

        for stopover in stopovers:
            self.set_position(stopover)
            self._sleep_simtime(0.2)

        self.set_orientation(offset_direction_quat)
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
        while self.t < current_timestep + n and self.jenga_env.simulation_running():
            time.sleep(0.05)


    def _sleep_simtime(self, t):
        current_time = self.t * g_timestep
        while self.t * g_timestep < current_time + t and self.jenga_env.simulation_running():
            time.sleep(0.05)

    @staticmethod
    def generate_xml():
        return f'''<body name="extractor" pos="{Extractor.HOME_POS[0]} {Extractor.HOME_POS[1]} {Extractor.HOME_POS[2]}" euler="0 0 0">
                            <joint name="extractor_slide_x" type="slide" pos="0 0 0" axis="1 0 0" damping="{Extractor.base_damping_slide}"/>
                            <joint name="extractor_slide_y" type="slide" pos="0 0 0" axis="0 1 0" damping="{Extractor.base_damping_slide}"/>
                            <joint name="extractor_slide_z" type="slide" pos="0 0 0" axis="0 0 1" damping="{Extractor.base_damping_slide}"/>
                            <joint name="extractor_hinge_x" type="hinge" axis="1 0 0" damping="{Extractor.base_damping_hinge}" pos ="0 0 0"/>
                            <joint name="extractor_hinge_y" type="hinge" axis="0 1 0" damping="{Extractor.base_damping_hinge}" pos ="0 0 0"/>
                            <joint name="extractor_hinge_z" type="hinge" axis="0 0 1" damping="{Extractor.base_damping_hinge}" pos ="0 0 0"/>

                            
                            <geom type="box" pos="0 0 0" size="{Extractor.size} {Extractor.width} {Extractor.size}" mass="{Extractor.base_mass}"/>
                            <body name="finger1" pos="{-Extractor.finger_length - Extractor.size} {-Extractor.width + Extractor.size} 0">
                                <joint name="finger1_joint" pos="0 0 0" type="slide" axis="0 1 0" damping="{Extractor.finger_damping}"/>
                                <geom type="box" size="{Extractor.finger_length} {Extractor.size} {Extractor.size}" mass="{Extractor.finger_mass}" friction="{1*5} {0.005*5} {0.0001*5}"/>
                            </body>
                            <body name="finger2" pos="{-Extractor.finger_length - Extractor.size} {Extractor.width - Extractor.size} 0">
                                <joint name="finger2_joint" pos="0 0 0" type="slide" axis="0 1 0" damping="{Extractor.finger_damping}"/>
                                <geom type="box" size="{Extractor.finger_length} {Extractor.size} {Extractor.size}" mass="{Extractor.finger_mass}" friction="{1*5} {0.005*5} {0.0001*5}"/>
                            </body>                            
                            <!-- <geom type="box" size="0.4 0.4 0.4" mass="0.1" rgba="1 0 0 1" pos="0 0 0"/> -->
                            
                            <!-- boxes for axes marking -->
                            <!-- <geom type="box" size="0.2 0.2 0.01" mass="0.1" rgba="0 1 0 1" pos="0 {1} {Extractor.size + 0.01}"/> -->
                        </body>'''
