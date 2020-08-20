from tower import Tower
import gym
from cv.camera import Camera
from constants import *
import dt_apriltags
from cv.blocks_calibration import read_corrections, read_block_sizes
from robots.robot import Robot, CoordinateSystem
import logging
import colorlog
import copy
import time
import matplotlib; matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import random
from robots.gripper import Gripper
from utils.utils import calculate_rotation, euler2quat
from utils.utils import Line

log = logging.Logger(__name__)
formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)sPID:%(process)d:%(funcName)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
log.addHandler(stream_handler)

class jenga_env(gym.Env):
    def __init__(self):
        # initialize cameras
        cam1 = Camera(cam1_serial, cam1_mtx, cam1_dist)
        cam2 = Camera(cam2_serial, cam2_mtx, cam2_dist)
        # cam1.start_grabbing()
        # cam2.start_grabbing()

        # get corrections
        corrections = read_corrections('./cv/corrections.json')

        # define block sizes
        self.block_sizes = read_block_sizes('./cv/block_sizes.json')
        detector = dt_apriltags.Detector(nthreads=detection_threads,
                                         quad_decimate=quad_decimate,
                                         quad_sigma=quad_sigma,
                                         decode_sharpening=decode_sharpening)

        # initialize tower
        self.tower = Tower(sim=None, viewer=None, simulation_fl=False,
                           cam1=cam1, cam2=cam2, at_detector=detector,
                           block_sizes=self.block_sizes, corrections=corrections)

        # initialize robot coordinate system
        x_ax = np.array([404.36, -91.24, 0.36])
        y_ax = np.array([331.34, 307.78, 1.09])
        origin = np.array([565.7, 65.05, 0.56])
        coord_system = CoordinateSystem.from_three_points(origin, x_ax, y_ax)


        # initialize the robot
        gripper = Gripper(right_gripper_ip)
        self.robot = Robot(right_robot_ip, right_robot_port, coord_system, gripper)
        self.robot.connect()

        print(f"Robot pose: {self.robot.get_world_pose(degrees=False)}")

        # initialize state variables
        self.total_distance = 0
        self.current_block_id = 0
        self.current_lvl = 0
        self.current_lvl_pos = 0
        self.initial_tilt_2ax = np.array([0, 0])
        self.previous_tilt_1ax = 0
        self.steps_pushed = 0
        self.max_pushing_distance = 25
        self.last_force = 0  # needed for calculating block displacement using force
        self.checked_positions = {i: {j for j in range(3)} for i in range(3, g_blocks_num - 9 + 3)}

        # reference force
        self.reference_force = 0

        # global parameters
        self.gap = 2  # in mm
        self.grip_depth_narrow = 25
        self.pull_distance = block_length_max - self.grip_depth_narrow + 10
        self.placing_vert_gap = 3  # in mm
        self.placing_pos_correction = np.array([-1.5, 0, 3])
        self.placing_quat_correction = Quaternion(axis=z_unit_vector, degrees=-0.5)

    def _get_stopovers_old(self, start_quarter, end_quarter, tower_xy, height):
        assert start_quarter in [1, 2, 3, 4] and end_quarter in [1, 2, 3, 4]
        distance_from_origin = block_length_mean * .8

        stopovers = []
        if start_quarter == 1 and end_quarter == 2:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(y_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 1 and end_quarter == 4:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(x_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 2 and end_quarter == 1:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(y_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 2 and end_quarter == 3:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-x_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 3 and end_quarter == 2:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-x_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 3 and end_quarter == 4:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-y_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 4 and end_quarter == 3:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-y_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 4 and end_quarter == 1:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(x_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 2 and end_quarter == 4:
            stopovers = self._get_stopovers(2, 3, tower_xy, height) + self._get_stopovers(3, 4, tower_xy, height)
        if start_quarter == 4 and end_quarter == 2:
            stopovers = self._get_stopovers(4, 1, tower_xy, height) + self._get_stopovers(1, 2, tower_xy, height)
        if start_quarter == 1 and end_quarter == 3:
            stopovers = self._get_stopovers(1, 2, tower_xy, height) + self._get_stopovers(2, 3, tower_xy, height)
        if start_quarter == 3 and end_quarter == 1:
            stopovers = self._get_stopovers(3, 4, tower_xy, height) + self._get_stopovers(4, 1, tower_xy, height)
        if start_quarter == end_quarter:
            stopovers = []

        return stopovers

    def _get_stopovers(self, start_quarter, end_quarter, tower_xy, height):
        assert start_quarter in [1, 2, 3, 4] and end_quarter in [1, 2, 3, 4]
        distance_from_origin = block_length_mean * 0.8

        stopovers = []
        if start_quarter == 1 and end_quarter == 2:  # prevent moving from 1 quarter to quarter 2 directly
            stopovers = self._get_stopovers(1, 4, tower_xy, height) + \
                        self._get_stopovers(4, 3, tower_xy, height) + \
                        self._get_stopovers(3, 2, tower_xy, height)
        if start_quarter == 1 and end_quarter == 4:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(x_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 2 and end_quarter == 1:  # prevent moving from 2 quarter to quarter 1 directly
            stopovers = self._get_stopovers(2, 3, tower_xy, height) + \
                        self._get_stopovers(3, 4, tower_xy, height) + \
                        self._get_stopovers(4, 1, tower_xy, height)
        if start_quarter == 2 and end_quarter == 3:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-x_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 3 and end_quarter == 2:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-x_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 3 and end_quarter == 4:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-y_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 4 and end_quarter == 3:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(-y_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 4 and end_quarter == 1:
            q = Quaternion(axis=z_unit_vector, degrees=45)
            stopover = q.rotate(x_unit_vector) * distance_from_origin
            stopover += tower_xy
            stopovers = [np.array(stopover) + np.array([0, 0, height])]
        if start_quarter == 2 and end_quarter == 4:
            stopovers = self._get_stopovers(2, 3, tower_xy, height) + self._get_stopovers(3, 4, tower_xy, height)
        if start_quarter == 4 and end_quarter == 2:
            stopovers = self._get_stopovers(4, 3, tower_xy, height) + self._get_stopovers(3, 2, tower_xy, height)
        if start_quarter == 1 and end_quarter == 3:
            stopovers = self._get_stopovers(1, 4, tower_xy, height) + self._get_stopovers(4, 3, tower_xy, height)
        if start_quarter == 3 and end_quarter == 1:
            stopovers = self._get_stopovers(3, 4, tower_xy, height) + self._get_stopovers(4, 1, tower_xy, height)
        if start_quarter == end_quarter:
            stopovers = []

        return stopovers

    @staticmethod
    def _get_quarter(pos, tower_center):
        assert pos.size == 3, "Argument 'pos' is not a 3d array!"
        x = pos[0] - tower_center[0]
        y = pos[1] - tower_center[1]
        if y >= abs(x):
            return 1
        if y <= -abs(x):
            return 3
        if x >= 0 and abs(y) < abs(x):
            return 4
        if x < 0 and abs(y) < abs(x):
            return 2

    def move_along_own_axis(self, axis, distance):  # distance can be negative
        assert axis == 'x' or axis == 'y' or axis == 'z', "Wrong axis!"

        if axis == 'x':
            unit_vector = x_unit_vector
        if axis == 'y':
            unit_vector = y_unit_vector
        if axis == 'z':
            unit_vector = z_unit_vector

        gripper_quat = self.robot.get_world_orientation()
        gripper_pos = self.robot.get_world_position()

        translation_direction = np.array(gripper_quat.rotate(unit_vector))
        end_point = np.array([gripper_pos[0], gripper_pos[1], gripper_pos[2]]) + translation_direction * distance
        self.robot.set_world_pos(end_point)

    def move_to_block_push(self, lvl, pos, poses):
        # get positions and orientations
        positions = self.tower.get_positions_from_poses(poses)
        orientations = self.tower.get_orientations_from_poses(poses)
        block_id = self.tower.get_block_id_from_pos(lvl, pos, positions, orientations, real_tag_pos)

        if block_id is not None:
           self.move_to_block_id_push(block_id, poses)
        else:
            log.warning(f"There is no block on position ({lvl}, {pos}).")

        return block_id

    def move_to_random_block(self):
        # get block poses
        poses = self.tower.get_poses_cv()
        positions = self.tower.get_positions_from_poses(poses)


        # get only legal block positions (check only the blocks that are below the highest full layer)
        layers = self.tower.get_layers(positions)
        if len(layers[-1]) == 3:
            offset_from_top = 1
        else:
            offset_from_top = 2
        max_lvl = len(layers) - offset_from_top
        available_lvls = {lvl: self.checked_positions[lvl] for lvl in self.checked_positions if lvl < max_lvl}

        if not available_lvls:
            return False

        # randomly choose one position
        lvl = random.choice(list(available_lvls.keys()))
        pos = random.sample(self.checked_positions[lvl], 1)[0]

        # update global variables
        self.current_lvl = lvl
        self.current_lvl_pos = pos

        # remove position and lvl if it has no unchecked positions
        self.checked_positions[lvl].remove(pos)
        if not self.checked_positions[lvl]:
            del self.checked_positions[lvl]

        block_id = self.move_to_block_push(lvl, pos, poses)

        return block_id, lvl, pos

    def move_to_block_id_old(self, block_id, positions, orientations):
        print(f"Move#1")

        self.robot.open_wide()

        if block_id is not None:
            # update current block id
            self.current_block_id = block_id

            # reset last reference positions and orientations
            self.tower.last_ref_positions = copy.deepcopy(positions)
            self.tower.last_ref_orientations = copy.deepcopy(orientations)

            # get orientation of the target block as a quaternion
            block_quat = Quaternion(orientations[block_id])
            block_pos = np.array(positions[block_id]) #+ np.array([0, 0, 50])

            # calculate pusher orientation
            first_block_end = block_pos + block_quat.rotate(x_unit_vector) * block_length_mean / 2
            second_block_end = block_pos + block_quat.rotate(-x_unit_vector) * block_length_mean / 2
            first_distance = np.linalg.norm(real_tag_pos - first_block_end)
            second_distance = np.linalg.norm(real_tag_pos - second_block_end)
            if first_distance < second_distance:
                offset_direction_quat = block_quat
            else:
                offset_direction_quat = block_quat * Quaternion(axis=[0, 0, 1], degrees=180)

            block_x_face_normal_vector = offset_direction_quat.rotate(x_unit_vector)
            target_x = block_pos[0] + (block_length_mean / 2 + self.gap) * block_x_face_normal_vector[0]
            target_y = block_pos[1] + (block_length_mean / 2 + self.gap) * block_x_face_normal_vector[1]
            target_z = block_pos[2] + (block_length_mean / 2 + self.gap) * block_x_face_normal_vector[2]

            target = np.array([target_x, target_y, target_z]) + np.array(block_x_face_normal_vector) * block_length_mean * 0.5

            # move pusher backwards to avoid collisions
            self.move_away_from_tower()

            # switch to the middle point of two fingers
            self.robot.switch_tool_two_fingers()

            tower_center = self.tower.get_center_xy(positions)

            # move through stopovers
            robot_pos = self.robot.get_world_position()
            print(f"Tower center: {tower_center}")
            print(f"Robot pos: {robot_pos}")
            start_quarter = self._get_quarter(robot_pos, tower_center)
            dst_quarter = self._get_quarter(target, tower_center)
            stopovers = self._get_stopovers(start_quarter,
                                            dst_quarter,
                                            tower_center,
                                            target[2])

            # move through stopovers
            for stopover in stopovers:
                print(f"Stopover: {stopover}")
                quat = self.orientation_towards_tower_center(stopover, tower_center)
                self.robot.set_world_pos_orientation(stopover, quat)
                time.sleep(1)

                # move the gripper along y-axis
                # magic code, no time to explain and no time for code it properly
                q1 = (start_quarter - 1) % 4
                q2 = (dst_quarter - 1) % 4
                if q1 > q2:
                    self.move_along_own_axis('y', block_length_mean*1.5)
                if q1 < q2:
                    self.move_along_own_axis('y', -block_length_mean*1.5)

            # compute robot gripper orientation
            gripper_quat = offset_direction_quat

            # move to intermediate target
            self.robot.set_world_pos_orientation(target, gripper_quat)

            # switch to the finger tip tool
            self.robot.switch_tool_sprung_finger_tip()

            # move to intermediate target with finger tip
            self.robot.set_world_pos(target)

            # move to the end target
            target = np.array([target_x, target_y, target_z])
            self.robot.set_world_pos(target)
            print(f"Target: {target}")

        else:
            log.warning(f"There is no block with id {block_id}.")

        print(f"Move#2")

    def move_to_block_id_push(self, block_id, poses):
        self.robot.open_wide()

        if block_id is not None:
            # update current block id
            self.current_block_id = block_id

            # get positions and orientations
            positions = self.tower.get_positions_from_poses(poses)
            orientations = self.tower.get_orientations_from_poses(poses)

            # reset last reference positions and orientations
            self.tower.last_ref_positions = copy.deepcopy(positions)
            self.tower.last_ref_orientations = copy.deepcopy(orientations)

            # get orientation of the target block as a quaternion
            block_quat = Quaternion(orientations[block_id])
            block_pos = np.array(positions[block_id]) #+ np.array([0, 0, 50])

            # calculate pusher orientation
            first_block_end = block_pos + block_quat.rotate(x_unit_vector) * block_length_mean / 2
            second_block_end = block_pos + block_quat.rotate(-x_unit_vector) * block_length_mean / 2
            first_distance = np.linalg.norm(real_tag_pos - first_block_end)
            second_distance = np.linalg.norm(real_tag_pos - second_block_end)
            if first_distance > second_distance:
                offset_direction_quat = block_quat
            else:
                offset_direction_quat = block_quat * Quaternion(axis=[0, 0, 1], degrees=180)

            block_x_face_normal_vector = offset_direction_quat.rotate(x_unit_vector)
            target_x = block_pos[0] + (block_length_mean / 2 + self.gap) * block_x_face_normal_vector[0]
            target_y = block_pos[1] + (block_length_mean / 2 + self.gap) * block_x_face_normal_vector[1]
            target_z = block_pos[2] + (block_length_mean / 2 + self.gap) * block_x_face_normal_vector[2]

            target = np.array([target_x, target_y, target_z]) + np.array(block_x_face_normal_vector) * block_length_mean * 0.5

            # move pusher backwards to avoid collisions
            self.move_away_from_tower()

            # switch to finger tip
            self.robot.switch_tool_sprung_finger_tip()

            tower_center = self.tower.get_center_xy(positions)

            # move through stopovers
            robot_pos = self.robot.get_world_position()
            start_quarter = self._get_quarter(robot_pos, tower_center)
            dst_quarter = self._get_quarter(target, tower_center)

            # compute robot gripper orientation
            gripper_quat = offset_direction_quat

            # switch to the finger tip tool
            self.robot.switch_tool_sprung_finger_tip()

            if start_quarter != dst_quarter:
                # move upwards
                current_pos = self.robot.get_world_position()
                intermediate_pos = self.set_pos_height(current_pos, stopover_height)
                self.robot.set_world_pos(intermediate_pos)

                # move over the intermediate target
                intermediate_pos = self.set_pos_height(target, stopover_height)
                self.robot.set_world_pos_orientation_mov(intermediate_pos, gripper_quat, pos_flags='(7, 0)')



            # move to intermediate target
            self.robot.set_world_pos_orientation_mov(target, gripper_quat, pos_flags='(7, 0)')

            # move to the end target
            target = np.array([target_x, target_y, target_z])
            self.robot.set_world_pos(target)
            print(f"Target: {target}")

            # wait until force stabilizes
            self.wait_force_stabilizing()

            # set reference force
            self.set_reference_force()
        else:
            log.warning(f"There is no block with id {block_id}.")

        print(f"Move#2")

    def extract(self, block_id):
        # open gripper
        self.robot.open_wide()
        # move to block
        self.move_to_block_id_extract(block_id)
        # close gripper
        self.robot.grip()
        # pull
        self.pull()

    def pull(self):
        self.move_along_own_axis('x', self.pull_distance)

    def move_to_block_id_extract(self, block_id):
        print(f"Move#1")
        if block_id is not None:
            # get poses
            poses = self.tower.get_poses_cv()
            positions = self.tower.get_positions_from_poses(poses)
            orientations = self.tower.get_orientations_from_poses(poses)

            # get orientation of the target block as a quaternion
            block_quat = Quaternion(orientations[block_id])
            block_pos = np.array(positions[block_id])

            # calculate gripper orientation
            first_block_end = block_pos + block_quat.rotate(x_unit_vector) * block_length_mean / 2
            second_block_end = block_pos + block_quat.rotate(-x_unit_vector) * block_length_mean / 2
            first_distance = np.linalg.norm(real_tag_pos - first_block_end)
            second_distance = np.linalg.norm(real_tag_pos - second_block_end)
            if first_distance < second_distance:
                offset_direction_quat = block_quat
            else:
                offset_direction_quat = block_quat * Quaternion(axis=[0, 0, 1], degrees=180)

            block_x_face_normal_vector = offset_direction_quat.rotate(x_unit_vector)
            target_x = block_pos[0] + (block_length_mean / 2 - self.grip_depth_narrow) * block_x_face_normal_vector[0]
            target_y = block_pos[1] + (block_length_mean / 2 - self.grip_depth_narrow) * block_x_face_normal_vector[1]
            target_z = block_pos[2] + (block_length_mean / 2 - self.grip_depth_narrow) * block_x_face_normal_vector[2] #+ 150

            target = np.array([target_x, target_y, target_z]) + np.array(
                block_x_face_normal_vector) * block_length_mean * 0.5

            # move pusher backwards to avoid collisions
            self.move_away_from_tower()

            # switch to the finger tip tool
            self.robot.switch_tool_two_fingers()

            # get tower center point
            tower_center = self.tower.get_center_xy(positions)

            # move through stopovers
            robot_pos = self.robot.get_world_position()
            start_quarter = self._get_quarter(robot_pos, tower_center)
            dst_quarter = self._get_quarter(target, tower_center)

            # compute robot gripper orientation
            gripper_quat = offset_direction_quat# * Quaternion(axis=z_unit_vector, degrees=180)

            if start_quarter != dst_quarter:
                # move upwards
                current_pos = self.robot.get_world_position()
                intermediate_pos = self.set_pos_height(current_pos, stopover_height)
                self.robot.set_world_pos(intermediate_pos)

                # move over the intermediate target
                intermediate_pos = self.set_pos_height(target, stopover_height)
                self.robot.set_world_pos_orientation_mov(intermediate_pos, gripper_quat, pos_flags='(7, 0)')

            # move to intermediate target
            self.robot.set_world_pos_orientation_mov(target, gripper_quat, pos_flags='(7, 0)')

            # move to the end target
            target = np.array([target_x, target_y, target_z])
            self.robot.set_world_pos(target)
            print(f"Target: {target}")

        else:
            log.warning(f"There is no block with id {block_id}.")

        print(f"Move#2")

    def push_through(self, distance):
        self.move_along_own_axis('x', -distance)

    def push(self):
        self.move_along_own_axis('x', -1)  # move 1mm
        self.wait_force_stabilizing()
        force = self.get_averaged_force(5) - self.reference_force
        block_displacement = self.calculate_block_displacement_from_force(1, force - self.last_force)
        self.last_force = force
        return force, block_displacement

    def calculate_block_displacement_from_force(self, robot_displacement, measured_force):
        block_displacement = robot_displacement - measured_force/right_robot_spring_constant

        # displacement cannot be negative
        block_displacement = max(0, block_displacement)

        return block_displacement

    def wait_force_stabilizing(self):
        time.sleep(read_force_wait)

    def move_away_from_tower(self):
        self.robot.switch_tool_sprung_finger_tip()
        distance1 = max(0, block_length_max - np.linalg.norm(
            self.tower.initial_pos[:2] - self.robot.get_world_position()[:2]))
        self.robot.switch_tool_two_fingers()
        distance2 = max(0, block_length_max - np.linalg.norm(
            self.tower.initial_pos[:2] - self.robot.get_world_position()[:2]))
        distance = max(distance1, distance2)
        self.move_along_own_axis('x', distance)

    def orientation_towards_tower_center(self, point, tower_center):
        point = np.copy(point)
        point[2] = 0
        tower_center = np.copy(tower_center)
        tower_center[2] = 0
        orientation_vec = tower_center - point
        orientation_vec /= np.linalg.norm(orientation_vec)
        quat = calculate_rotation(x_unit_vector, orientation_vec)
        return quat

    def set_pos_height(self, pos, height):
        res = copy.deepcopy(pos)
        res[2] = height
        return res

    def go_home_old(self):
        print(f"Go home#1")

        print(f"Before move away")
        self.move_away_from_tower()
        print(f"After move away")

        # swith to the right tool
        self.robot.switch_tool_taster()

        # get tower center
        tower_center = self.tower.initial_pos

        # move through stopovers
        start_quarter = self._get_quarter(self.robot.get_world_position(), tower_center)
        dst_quarter = self._get_quarter(right_robot_home_position_world[:3], tower_center)
        # stopovers = self._get_stopovers(start_quarter,
        #                                 dst_quarter,
        #                                 tower_center,
        #                                 right_robot_home_position_world[2])

        # move through stopovers
        # print(f"Before move through stopovers")
        # for stopover in stopovers:
        #     quat = self.orientation_towards_tower_center(stopover, tower_center)
        #     self.robot.set_world_pos_orientation(stopover, quat)
        #     time.sleep(1)
        #
        #     # move the gripper along y-axis
        #     # magic code, no time to explain and no time for code it properly
        #     q1 = (start_quarter - 1) % 4
        #     q2 = (dst_quarter - 1) % 4
        #     if q1 > q2:
        #         self.move_along_own_axis('y', block_length_mean*1.5)
        #     if q1 < q2:
        #         self.move_along_own_axis('y', -block_length_mean*1.5)
        # print(f"After move through stopovers")

        if start_quarter != dst_quarter:
            # move upwards
            current_pos = self.robot.get_world_position()
            intermediate_pos = self.set_pos_height(current_pos, stopover_height)
            self.robot.set_world_pos(intermediate_pos)

            home_pose = copy.deepcopy(right_robot_home_position_world)
            home_pose[2] = stopover_height
            self.robot.set_world_pose(home_pose, degrees=True)

        # move to the end target
        self.robot.set_world_pose(right_robot_home_position_world, degrees=True)

        print(f"Go home#2")

    def go_home(self):
        print(f"Go home#1")

        print(f"Before move away")
        self.move_away_from_tower()
        print(f"After move away")

        # swith to the right tool
        self.robot.switch_tool_two_fingers()

        # move upwards
        self.move_upwards(stopover_height)

        # get tower center
        tower_center = self.tower.initial_pos

        # move through stopovers
        start_quarter = self._get_quarter(self.robot.get_world_position(), tower_center)
        dst_quarter = self._get_quarter(right_robot_home_position_world[:3], tower_center)

        if start_quarter != dst_quarter:
            # move upwards
            current_pos = self.robot.get_world_position()
            intermediate_pos = self.set_pos_height(current_pos, stopover_height)
            self.robot.set_world_pos(intermediate_pos)

            print(f"Moved upwards")

            # move over the destination position
            home_pose = copy.deepcopy(right_robot_home_position_world)
            home_pose[2] = stopover_height
            self.robot.set_world_pose_mov(home_pose, degrees=True, pos_flags='(7, 0)')

            print(f"Moved over destination")

        # move to the end target
        self.robot.set_world_pose_mov(right_robot_home_position_world, degrees=True, pos_flags='(7, 0)')

        print(f"Go home#2")

    def get_state(self, new_block, force=0, block_displacement=0, mode=0):
        """
        mode=0: return gym compatible state
        mode=1: return state as State class
        """

        if mode == 0:

            log.debug(f"Get state#1")

            # calculate side
            side = None
            if self.current_lvl % 2 == 0:
                side = 0
            else:
                side = 1

            # get block positions
            log.debug(f"Before get positions")
            poses = self.tower.get_poses_cv()
            block_positions = self.tower.get_positions_from_poses(poses)
            block_orientations = self.tower.get_orientations_from_poses(poses)
            log.debug(f"After get positions")

            # calculate tilt
            log.debug(f"Before get tilt")
            tilt_2ax = self.tower.get_tilt_2ax(block_positions, block_orientations, self.current_block_id)
            log.debug(f"After get tilt")
            if new_block:
                self.initial_tilt_2ax = tilt_2ax
                self.previous_step_displacement = np.array([0, 0])
                self.previous_step_z_rot = 0
            last_tilt_2ax = tilt_2ax - self.initial_tilt_2ax

            # get last z_rotation
            log.debug(f"Before get z rotaiton")
            z_rotation_last = self.tower.get_last_z_rotation(self.current_block_id, block_orientations)
            log.debug(f"After get z rotaiton")

            # get current round displacement
            current_round_displacement = self.tower.get_last_displacement_2ax(self.current_block_id, block_positions, block_orientations)
            log.debug(f"After get displacement")

            # get last step tower displacement and z rotation
            current_step_tower_displacement = current_round_displacement - self.previous_step_displacement
            current_step_z_rot = z_rotation_last - self.previous_step_z_rot

            # reset previous displacement and z rotation
            self.previous_step_displacement = current_round_displacement
            self.previous_step_z_rot = z_rotation_last

            total_block_distance = self.total_distance

            # get block height
            block_height = block_positions[self.current_block_id][2] / one_millimeter

            # get layer configuration
            # 0) □□□
            # 1) □x□
            # 2) □□x
            # 3) x□□
            # 4) x□x
            log.debug(f"Before get layers state")
            layers = self.tower.get_layers_state(block_positions, block_orientations, real_tag_pos)
            log.debug(f"After get layers state")
            current_layer = layers[self.current_lvl]
            print(f"Current layer: {current_layer}")
            layer_configuration = 0
            if current_layer[0] is not None and current_layer[1] is not None and current_layer[2] is not None:
                layer_configuration = 0
            elif current_layer[0] is not None and current_layer[2] is not None:
                layer_configuration = 1
            elif current_layer[0] is not None and current_layer[1] is not None:
                layer_configuration = 2
            elif current_layer[1] is not None and current_layer[2] is not None:
                layer_configuration = 3
            elif current_layer[1] is not None:
                layer_configuration = 4

            # forma a state
            state = np.array([force, block_displacement, total_block_distance, current_step_tower_displacement[0],
                             current_step_tower_displacement[1], current_round_displacement[0],
                             current_round_displacement[1], last_tilt_2ax[0], last_tilt_2ax[1], current_step_z_rot,
                             z_rotation_last, side, block_height, self.current_lvl_pos, layer_configuration])
            log.debug(f"Get state#2")
            return state

    # measures and averages forces to get the reference force
    def set_reference_force(self):
        self.reference_force = self.get_averaged_force(10)

    def get_averaged_force(self, n=5):
        forces = []
        for i in range(n):
            force = -self.robot.get_force()[0]
            forces.append(force)

        return np.mean(forces)

    def move_upwards(self, height):
        current_pos = self.robot.get_world_position()
        intermediate_pos = self.set_pos_height(current_pos, height)
        self.robot.set_world_pos(intermediate_pos)

    def place_on_zwischenablage(self):
        # move upwards
        self.move_upwards(stopover_height)

        # move over the destination position
        intermediate_pos = copy.deepcopy(zwischenablage_place_pose[:3])
        intermediate_pos[2] = stopover_height
        self.robot.set_world_pos_orientation_mov(intermediate_pos, zwischenablage_place_quat, pos_flags='(7, 0)')

        # move to the end target
        self.robot.set_world_pose_mov(zwischenablage_place_pose, degrees=True, pos_flags='(7, 0)')

        # release block
        self.robot.open_wide()

        # move upwards
        self.move_upwards(150)

    def take_from_zwischenablage(self):
        # move to intermediate pos
        intermediate_pos = zwischenablage_take_pos + zwischenablage_take_quat.rotate(x_unit_vector) * 50
        self.robot.set_world_pos_orientation(intermediate_pos, zwischenablage_take_quat)

        # move to the final target
        self.robot.set_world_pose(zwischenablage_take_pose, degrees=True)

        # grip
        self.robot.grip()

        # move upwards
        current_pos = self.robot.get_world_position()
        waiting_pos = current_pos
        waiting_pos[2] = stopover_height
        self.robot.set_world_pos_orientation(waiting_pos, Quaternion(axis=z_unit_vector, degrees=45))

    def calculate_gripper_orientation(self, block_orientation):
        dot_x = np.dot(block_orientation.rotate(x_unit_vector), x_unit_vector)
        dot_y = np.dot(block_orientation.rotate(x_unit_vector), y_unit_vector)

        orientation1 = block_orientation * Quaternion(axis=z_unit_vector, degrees=90)
        orientation2 = block_orientation * Quaternion(axis=z_unit_vector, degrees=-90)
        if abs(dot_x) >= 0.9:
            if np.dot(orientation1.rotate(x_unit_vector), y_unit_vector) < 0:
                return orientation1
            if np.dot(orientation2.rotate(x_unit_vector), y_unit_vector) < 0:
                return orientation2
        if abs(dot_y) >= 0.9:
            if np.dot(orientation1.rotate(x_unit_vector), x_unit_vector) > 0:
                return orientation1
            if np.dot(orientation2.rotate(x_unit_vector), x_unit_vector) > 0:
                return orientation2

    def place_on_top(self, block_id):
        # get block poses
        poses = self.tower.get_poses_cv()
        positions = self.tower.get_positions_from_poses(poses)
        orientations = self.tower.get_orientations_from_poses(poses)

        # get placing position and orientation
        pose_info = self.tower.get_placing_pose(positions, orientations, current_block=block_id)
        block_orientation = pose_info['orientation']

        # calculate gripper orientation
        gripper_orientation = self.calculate_gripper_orientation(block_orientation)

        pos_with_tolerance = pose_info['pos_with_tolerance']
        pos = pose_info['pos'] + gripper_orientation.rotate(self.placing_pos_correction)  # + np.array([0, 0, 100])
        last_stopover = pose_info['stopover'] + gripper_orientation.rotate(self.placing_pos_correction)  # + np.array([0, 0, 100])

        # correct gripper
        gripper_orientation = gripper_orientation * self.placing_quat_correction

        # go over the last stopover
        intermediate_pos = self.set_pos_height(last_stopover, stopover_height)
        self.robot.set_world_pos_orientation_mov(intermediate_pos, gripper_orientation, pos_flags='(7, 0)')

        # go to the last stopover near the tower
        self.robot.set_world_pos_orientation_mov(last_stopover, gripper_orientation, pos_flags='(7, 0)')

        # move to the target position
        self.robot.set_world_pos(pos, speed=1)

        # place
        self.robot.release()

        # open wide
        self.robot.open_wide()

        # move upwards
        self.move_upwards(self.robot.get_world_position()[2] + 100)

    def extract_and_place_on_top(self, block_id):
        # extract block
        self.extract(block_id)

        # put on zwischenablage
        self.place_on_zwischenablage()

        # take from zwischenablage
        self.take_from_zwischenablage()

        # place on top
        self.place_on_top(block_id)

        # go home
        self.go_home()

    def check_blocks(self):
        self.go_home()

        for j in range(30):
            block_id, lvl, pos = self.move_to_random_block()
            for i in range(30):
                force, displacement = self.push()
                print(f"Step #i: Force: {force}, displacement: {displacement}")
                if force > self.get_force_threshold(lvl):
                    break
                if i > 10:
                    self.push_through(20)
                    self.extract_and_place_on_top(block_id)
                    break

    def check_extractable_blocks(self):
        self.go_home()

        loose_blocks = self.find_extractable_blocks()
        for block_id, lvl, pos in loose_blocks:
            poses = self.tower.get_poses_cv()
            self.move_to_block_id_push(block_id, poses)
            for i in range(30):
                force, displacement = self.push()
                print(f"Step #i: Force: {force}, displacement: {displacement}")
                if force > self.get_force_threshold(lvl):
                    break
                if i > 10:
                    self.push_through(20)
                    self.extract_and_place_on_top(block_id)
                    break

    def get_force_threshold(self, lvl):
        max = 0.7
        min = 0.1
        max_lvl = 17
        return max - ((max-min)/max_lvl) * lvl


    # debugging funciton
    def debug_push(self):
        self.move_along_own_axis('x', -1)  # move 1mm
        start = time.time()
        for i in range(100):
            force = -self.robot.get_force()[0] - self.reference_force
            print(f"Force: {force}, elapsed: {time.time() - start:.3f}")

        self.wait_force_stabilizing()
        force = self.get_averaged_force(5) - self.reference_force
        block_displacement = self.calculate_block_displacement_from_force(1, force - self.last_force)
        self.last_force = force
        return force, block_displacement

    def find_extractable_blocks(self):
        poses = self.tower.get_poses_cv()
        positions = self.tower.get_positions_from_poses(poses)
        orientations = self.tower.get_orientations_from_poses(poses)
        layers = self.tower.get_layers_state(positions, orientations, origin)

        loose_blocks = []
        print(f"Layers: {layers}")
        for i in layers:
            if i > 2:
                id0 = layers[i][0]
                id1 = layers[i][1]
                id2 = layers[i][2]
                if id0 is not None and id1 is not None and id2 is not None:
                    height0 = self.block_sizes[id0]['height']
                    height1 = self.block_sizes[id1]['height']
                    height2 = self.block_sizes[id2]['height']

                    if i == 9:
                        print(f"Block #{id0}: {height0}")
                        print(f"Block #{id1}: {height1}")
                        print(f"Block #{id2}: {height2}")
                    if i == 12:
                        print(f"Block #{id0}: {height0}")
                        print(f"Block #{id1}: {height1}")
                        print(f"Block #{id2}: {height2}")

                    # define points
                    p0 = (0, height0)
                    p1 = (block_width_mean, height0)
                    p2 = (block_width_mean, height1)
                    p3 = (2*block_width_mean, height1)
                    p4 = (2*block_width_mean, height2)
                    p5 = (3*block_width_mean, height2)

                    if height1 - height0 >= loose_block_height_threshold and height1 - height2 >= loose_block_height_threshold:
                        loose_blocks.append((id0, i, 0))
                        loose_blocks.append((id2, i, 2))
                    elif height0 > height2:
                        line = Line(p1, p5)
                        if line.f(p2[0]) - p2[1] >= loose_block_height_threshold and line.f(p3[0]) - p3[1] >= loose_block_height_threshold:
                            loose_blocks.append((id1, i, 1))
                        elif p2[1] - line.f(p2[0]) >= loose_block_height_threshold or p3[1] - line.f(p3[0]) >= loose_block_height_threshold:
                            loose_blocks.append((id2, i, 2))
                    else:
                        line = Line(p0, p4)
                        if line.f(p2[0]) - p2[1] >= loose_block_height_threshold and line.f(p3[0]) - p3[
                            1] >= loose_block_height_threshold:
                            loose_blocks.append((id1, i, 1))
                        elif p2[1] - line.f(p2[0]) >= loose_block_height_threshold or p3[1] - line.f(
                                p3[0]) >= loose_block_height_threshold:
                            loose_blocks.append((id0, i, 0))
                elif id0 is not None and id1 is not None:
                    height0 = self.block_sizes[id0]['height']
                    height1 = self.block_sizes[id1]['height']

                    if height1 - height0 >= loose_block_height_threshold:
                        loose_blocks.append((id0, i, 0))
                elif id2 is not None and id1 is not None:
                    height1 = self.block_sizes[id1]['height']
                    height2 = self.block_sizes[id2]['height']

                    if height1 - height2 >= loose_block_height_threshold:
                        loose_blocks.append((id2, i, 2))

        return loose_blocks




if __name__ == "__main__":
    jenga = jenga_env()

    blocks = [(0,1), (1, 2), (2, 2), (3, 0), (3, 2), (4, 1), (5, 2), (6, 2), (7, 2), (8, 2), (9, 1), (10, 1), (11, 0), (11, 2), (13, 1)]

    #
    # tower_center = jenga.tower.get_center_xy(positions)
    #
    # # move through stopovers
    # quarter1 = jenga._get_quarter(jenga.robot.get_world_position(), tower_center)
    # print(f"Quarter: {quarter1}")
    #
    # exit(0)

    print(jenga.find_extractable_blocks())
    print(f"State:")
    print(f"{jenga.get_state(new_block=True)}")

    # jenga.check_extractable_blocks()

    exit()

    jenga.go_home()

    for i in range(10):
        res = jenga.move_to_random_block(positions)
        print(f"Result of move_to_random_block: {res}")
    exit()


    for i in range(6):
        jenga.go_home()
        jenga.take_from_zwischenablage()
        poses = jenga.tower.get_poses_cv()
        positions = jenga.tower.get_positions_from_poses(poses)
        orientations = jenga.tower.get_orientations_from_poses(poses)
        jenga.place_on_top(47, positions, orientations)
        poses = jenga.tower.get_poses_cv()
        positions = jenga.tower.get_positions_from_poses(poses)
        orientations = jenga.tower.get_orientations_from_poses(poses)
        jenga.tower.calculate_orientation_corrections(orientations)

    exit()

    for i in range(3):
        print(f"For go home")
        jenga.go_home()
        print(f"After go home")

        jenga.extract(11, positions, orientations)
        jenga.place_on_zwischenablage()
        jenga.take_from_zwischenablage()
        jenga.place_on_top(11, positions, orientations)


    exit()


    poses = jenga.tower.get_poses_cv()
    print(f"Poses: {poses}")
    positions = jenga.tower.get_positions_from_poses(poses)
    orientations = jenga.tower.get_orientations_from_poses(poses)



    # jenga.move_to_block(7, 1)
    # time.sleep(1)
    # jenga.set_reference_force()
    # jenga.push()

    lvls = [i for i in range(0, 18, 1) if i % 2 == 1]
    for i in range(15):
        lvl = random.choice(lvls)
        pos = int(random.random() * 3)
        block_id = jenga.tower.get_block_id_from_pos(lvl, pos, positions, orientations, real_tag_pos)
        jenga.move_to_block_id_push(block_id, positions, orientations)
    exit(0)
    forces = dict()
    for i in range(10):
        lvls = [i for i in range(0, 18, 1) if i % 2 == 1]
        lvl = random.choice(lvls)
        pos = int(random.random()*3)
        block_id = jenga.tower.get_block_id_from_pos(lvl, pos, positions, orientations, real_tag_pos)
        jenga.move_to_block_id_push(block_id, positions, orientations)
        jenga.wait_force_stabilizing()

        forces[block_id] = []
        jenga.set_reference_force()
        for i in range(35):
            force = jenga.push()
            forces[block_id].append(force)
            if force > 0.15:
                break
            if i > 10:
                jenga.push_through(35 - 10)
                break
        jenga.go_home()

    print(repr(forces))


    jenga.go_home()