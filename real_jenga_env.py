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
import matplotlib.animation as animation
style.use('fivethirtyeight')
import random
from robots.gripper import Gripper
from utils.utils import calculate_rotation, euler2quat
from utils.utils import Line
from multiprocessing import Process, Queue
import traceback
from tensorflow import keras
from tensorflow.keras.layers import Dense, BatchNormalization

log = logging.Logger(__name__)
formatter = colorlog.ColoredFormatter('%(log_color)s%(levelname)sPID:%(process)d:%(funcName)s:%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.DEBUG)
log.addHandler(stream_handler)

class jenga_env(gym.Env):
    action_space = gym.spaces.Discrete(2)
    high = np.array([0.9, 13.86666667, 32.13333333, 1.16353707,
                     2.09707612, 7.80773817, 299.99928911])
                    # , 2.,
                    #  4.])
    low = np.array([-0.056, 0., 0., 0., 0.,
                    0., 52.84550074])
                    # 0., 0.])
    observation_space = gym.spaces.Box(low=high, high=high)

    def __init__(self, normalize):
        # initialize cameras
        self.cam1 = Camera(cam1_serial, cam1_mtx, cam1_dist)
        self.cam2 = Camera(cam2_serial, cam2_mtx, cam2_dist)

        assert one_millimeter == 1, "Wrong scaler parameter!"

        # get corrections
        self.corrections = read_corrections('./data/corrections.json')

        # define block sizes
        self.block_sizes = read_block_sizes('./data/block_sizes.json')
        self.detector1 = dt_apriltags.Detector(nthreads=detection_threads,
                                         quad_decimate=quad_decimate1,
                                         quad_sigma=quad_sigma1,
                                         decode_sharpening=decode_sharpening1)
        self.detector2 = dt_apriltags.Detector(nthreads=detection_threads,
                                               quad_decimate=quad_decimate2,
                                               quad_sigma=quad_sigma2,
                                               decode_sharpening=decode_sharpening2)

        # initialize robot coordinate system
        coord_system = CoordinateSystem.from_three_points(right_robot_origin_point,
                                                          right_robot_x_axis_point,
                                                          right_robot_y_axis_point)

        # initialize the robot
        gripper = Gripper(right_gripper_ip)
        self.robot = Robot(right_robot_ip, right_robot_port, coord_system, gripper)
        self.robot.connect()

        self.normalize = normalize

        self.initialize_global_variables()

    def initialize_global_variables(self):
        # initialize tower
        self.tower = Tower(sim=None, viewer=None, simulation_fl=False,
                           cam1=self.cam1, cam2=self.cam2, at_detector1=self.detector1,
                           at_detector2=self.detector2,
                           block_sizes=self.block_sizes, corrections=self.corrections)

        # initialize state variables
        self.total_distance = 0
        self.current_block_id = 0
        self.current_lvl = 0
        self.current_lvl_pos = 0
        self.initial_tilt_2ax = np.array([0, 0])
        self.previous_tilt_1ax = 0
        self.steps_pushed = 0
        self.max_pushing_distance = 30
        self.individual_pushing_distance = 10
        self.last_force = 0  # needed for calculating block displacement using force
        # self.positions_to_check = {i: {j for j in range(3)} for i in range(4, g_blocks_num - 9 + 3)}
        # self.positions_to_check = {i: {j for j in range(3)} for i in range(6, g_blocks_num - 9 + 3)}
        self.positions_to_check = self.get_blocks_to_check_ml()
        # self.checked_positions = {3: {0, 1, 2}, 7: {0, 1, 2}, 12: {0, 1, 2}, 11: {0, 1, 2}, }
        # self.checked_positions = {4: {0, 1, 2}, 5: {0, 1, 2}, 6: {1, 2}, 7: {1, 2}, 8: {0, 1, 2}, 9: {1}, 10: {0, 1, 2}, 11: {1}, 12: {1}, 13: {0, 2}, 14: {1}, 15: {0, 1, 2}, 16: {1}, 17: {0, 2}}
        # self.positions_to_check = self.create_random_blocks_to_check(30)



        # reference force
        self.reference_force = 0

        # global parameters
        self.gap = 2  # in mm
        self.grip_depth_narrow = 25
        self.pull_distance = block_length_max - self.grip_depth_narrow + 10
        self.placing_vert_gap = 3  # in mm
        self.placing_pos_correction = np.array([-1.5, 0, 3])
        self.placing_quat_correction = Quaternion(axis=z_unit_vector, degrees=-0.5)
        self.pushing_distance = 2  # in mm

        # globals
        self.last_screenshot = None
        self.last_state = None
        self.last_reward = None
        self.previous_step_displacement = None
        self.previous_step_max_displacement = None
        self.previous_step_z_rot = None
        self.extracted_blocks = 0
        self.plot_state_data = True
        self.positions_from_last_step = copy.deepcopy(self.tower.ref_positions)
        self.orientations_from_last_step = copy.deepcopy(self.tower.ref_orientations)
        self.blocks_for_features = set([i for i in range(g_blocks_max)])

        if self.plot_state_data:
            self.g_state_data_queue = []
            self.g_state_data_queue_maxsize = 250
            self.env_state_data_q = Queue()
            state_plotting_process = Process(target=self.plot_env_state_data, args=(self.env_state_data_q,))
            state_plotting_process.start()

    def create_random_blocks_to_check(self, n):
        max_lvl = len(
            self.tower.get_layers_state(self.tower.last_ref_positions, self.tower.last_ref_orientations, origin))
        i = 0
        positions_to_check = dict()
        while i < n:
            lvl = 4 + int(random.random() * (max_lvl - 3))
            pos = int(random.random() * 3)
            exists = False
            if lvl in positions_to_check and pos in positions_to_check[lvl]:
                exists = True

            if not exists:
                i += 1
                if lvl in positions_to_check:
                    positions_to_check[lvl].add(pos)
                else:
                    positions_to_check[lvl] = {pos}

        return positions_to_check

    def plot_env_state_data(self, env_state_data_q):
        def animate_state_data(i):
            if not env_state_data_q.empty():
                data = env_state_data_q.get()
            else:
                return

            states = [i[0] for i in data]
            rewards = [i[1] for i in data]

            states = states[1:]
            rewards = rewards[1:]

            forces = [i[0] for i in states]
            block_displacements = [i[1] for i in states]
            total_displacements = [i[2] for i in states]
            current_step_tower_disp = [i[3] for i in states]
            current_round_tower_disp = [i[4] for i in states]
            current_step_max_disp = [i[5] for i in states]
            block_height = [i[6] for i in states]

            ys = []
            ys.append(current_step_tower_disp)
            ys.append(current_step_max_disp)
            ys.append(block_displacements)
            ys.append(total_displacements)
            ys.append(forces)
            ys.append(rewards)

            axes_titles = ['Current step average displacement', 'Current step max displacement', 'Block displacement', 'Total block displacement', 'Force', 'Reward']


            y_limits = []
            y_limits.append([-1, 3])
            y_limits.append([-1, 3])
            y_limits.append([-1, 3])
            y_limits.append([0, 25])
            y_limits.append([-0.3, 0.7])
            y_limits.append([-5, 5])
            y_limits.append([-0.5, 0.5])

            # x-data
            xs = range(0, len(forces))

            # update plots

            for i in range(n):
                axs[i].clear()
                axs[i].set_ylim(y_limits[i])
                axs[i].title.set_text(axes_titles[i])
                axs[i].plot(xs, ys[i])

        # define plot for the force data plotting
        n = 6
        state_figure, axs = plt.subplots(nrows=3, ncols=2, num='Behavioral Cloning Features')
        axs = np.reshape(axs, (6,))

        # state_figure = plt.figure()
        # axes = []
        # for i in range(1, rows_n + 1):
        #     ax = state_figure.add_subplot(rows_n, 2, i)
        #     axes.append(ax)
        state_figure.tight_layout()

        ani = animation.FuncAnimation(state_figure, animate_state_data, interval=1)
        plt.show()

    def update_env_state_plot(self, state, reward):
        if len(self.g_state_data_queue) >= self.g_state_data_queue_maxsize:
            self.g_state_data_queue.pop(0)
        self.g_state_data_queue.append((state, reward))
        if not self.env_state_data_q.empty():
            # self.force_data_q.get()
            self.env_state_data_q.put(self.g_state_data_queue)
        else:
            self.env_state_data_q.put(self.g_state_data_queue)

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

    def move_along_own_axis(self, axis, distance, speed=0):  # distance can be negative
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
        self.robot.set_world_pos(end_point, speed=speed)

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

    def get_blocks_to_check_ml(self):
        blocks_to_check = {i: set() for i in range(3, g_blocks_num - 9 + 3)}

        input_shape = 6

        model = keras.Sequential()
        model.add(Dense(64, input_dim=input_shape, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.load_weights('./models/real_robot/loose_blocks_finding/loose_blocks_model_12_10_2020')

        block_sizes = read_block_sizes('./data/block_sizes.json')

        poses = self.tower.get_poses_cv()
        positions = self.tower.get_positions_from_poses(poses)
        orientations = self.tower.get_orientations_from_poses(poses)
        layers = self.tower.get_layers_state(positions, orientations, origin)

        print(f"Layers: {layers}")


        for lvl in layers:
            if lvl > 2:
                inverted = False
                # heights = [block_sizes[block_id]['height'] for block_id in layer]
                heights = []
                layer = [layers[lvl][i] for i in range(3)]
                absent_blocks = set()

                for i in range(len(layer)):
                    block_id = layer[i]
                    if block_id is not None:
                        heights.append(block_sizes[block_id]['height'])
                    else:
                        heights.append(block_height_min)
                        absent_blocks.update({i})
                if heights[0] > heights[2]:
                    heights[0], heights[2] = heights[2], heights[0]
                    inverted = True
                heights = np.reshape(heights, (1, 3))
                for pos, block_id in enumerate(layer):
                    zero_vec = np.zeros((1, 3))
                    zero_vec[0][pos] = 1
                    one_hot = zero_vec

                    heights_normalized = (heights - block_selection_mean[:3]) / block_selection_std[:3]

                    feature_vec = np.concatenate((heights_normalized, one_hot), axis=1)

                    prediction = model.predict(x=feature_vec)

                    if prediction > 0.2:
                        prediction = True
                    else:
                        prediction = False

                    if prediction:
                        if inverted:
                            if pos == 2:
                                if not 0 in absent_blocks:
                                    blocks_to_check[lvl].update({0})
                            elif pos == 0:
                                if not 2 in absent_blocks:
                                    blocks_to_check[lvl].update({2})
                            else:
                                if not pos in absent_blocks:
                                    blocks_to_check[lvl].update({pos})
                        else:
                            if not pos in absent_blocks:
                                blocks_to_check[lvl].update({pos})

                    # print(f"Feature vector: {feature_vec}")
                    # print(f"Predictions: {prediction}")


        #
        #
        # layer = [10, 9, 5]
        # heights = [block_sizes[i]['height'] for i in layer]
        # heights = [14.63,       14.94,       15.03]
        # heights = [block_sizes[13]['height'], block_sizes[17]['height'], block_sizes[6]['height']]
        # heights = np.array(heights)
        # heights = (heights - block_selection_mean[:3]) / block_selection_std[:3]
        # heights = np.reshape(heights, (1, 3))
        # zero_vec = np.zeros((1, 3))
        # zero_vec[0][1] = 1
        # feature_vec = np.concatenate((heights, zero_vec), axis=1)
        #
        # print(f"Feature vector: {feature_vec}")
        #
        # np.reshape(feature_vec, (1, 6))
        #
        # print(f"Predict: {model.predict(feature_vec)}")
        #
        # exit()
        remove = [k for k in blocks_to_check if blocks_to_check[k] == {} or blocks_to_check[k] == set()]
        for lvl in remove:
            del blocks_to_check[lvl]

        print(f"Blocks to check: {blocks_to_check}")

        return blocks_to_check

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
        available_lvls = {lvl: self.positions_to_check[lvl] for lvl in self.positions_to_check if lvl < max_lvl}

        if not available_lvls:
            print(f"No available lvls!")
            return None, None, None

        # randomly choose one position
        lvl = random.choice(list(available_lvls.keys()))
        pos = random.sample(self.positions_to_check[lvl], 1)[0]

        # update global variables
        self.current_lvl = lvl
        self.current_lvl_pos = pos

        # remove position and lvl if it has no unchecked positions
        self.positions_to_check[lvl].remove(pos)
        if not self.positions_to_check[lvl]:
            del self.positions_to_check[lvl]

        block_id = self.move_to_block_push(lvl, pos, poses)

        if block_id is None:
            print(f"Block_id is None!")
            block_id, lvl, pos = self.move_to_random_block()

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
            self.robot.switch_tool_sprung_finger_tip_right()

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
            self.move_away_from_tower(speed=10)

            # compute robot gripper orientation
            gripper_quat = offset_direction_quat

            # switch to finger tip
            dot_x = np.dot(x_unit_vector, gripper_quat.rotate(x_unit_vector))
            dot_y = np.dot(y_unit_vector, gripper_quat.rotate(x_unit_vector))
            print(f"Dot_x: {dot_x}")
            print(f"Dot_y: {dot_y}")
            if abs(dot_x) > abs(dot_y):
                self.robot.switch_tool_sprung_finger_tip_right()
            else:
                self.robot.switch_tool_sprung_finger_tip_left()

            tower_center = self.tower.get_center_xy(positions)

            # move through stopovers
            robot_pos = self.robot.get_world_position()
            start_quarter = self._get_quarter(robot_pos, tower_center)
            dst_quarter = self._get_quarter(target, tower_center)

            # if start_quarter != dst_quarter:

            # move upwards
            current_pos = self.robot.get_world_position()
            intermediate_pos = self.set_pos_height(current_pos, stopover_height)
            self.robot.set_world_pos(intermediate_pos, speed=10)

            # move over the intermediate target
            intermediate_pos = self.set_pos_height(target, stopover_height)
            self.robot.set_world_pos_orientation_mov(intermediate_pos, gripper_quat, pos_flags='(7, 0)', speed=10)



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
            self.move_away_from_tower(speed=10)

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
                self.robot.set_world_pos(intermediate_pos, speed=10)

                # move over the intermediate target
                intermediate_pos = self.set_pos_height(target, stopover_height)
                self.robot.set_world_pos_orientation_mov(intermediate_pos, gripper_quat, pos_flags='(7, 0)', speed=10)

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
        self.move_along_own_axis('x', -self.pushing_distance)

        # wait for tower stabilize
        start = time.time()
        time.sleep(0.2)
        poses = self.tower.get_poses_cv()
        # poses = None
        elapsed_poses = time.time() - start

        # wait remaining time
        time.sleep(max(0, read_force_wait - elapsed_poses))

        # self.wait_force_stabilizing()

        force = self.get_averaged_force(5) - self.reference_force
        print(f"Last force: {self.last_force}, Force: {force}, Force - last_force: {force}")
        block_displacement = self.calculate_block_displacement_from_force(self.pushing_distance, force)
        # self.last_force = force
        self.total_distance += block_displacement
        return force, block_displacement, poses

    def calculate_block_displacement_from_force(self, robot_displacement, measured_force):
        block_displacement = robot_displacement - measured_force/right_robot_spring_constant

        # displacement cannot be negative
        block_displacement = max(0, block_displacement)

        return block_displacement

    def wait_force_stabilizing(self):
        time.sleep(read_force_wait)

    def move_away_from_tower(self, speed=0):
        self.robot.switch_tool_sprung_finger_tip_right()
        distance1 = max(0, block_length_max - np.linalg.norm(
            self.tower.initial_pos[:2] - self.robot.get_world_position()[:2]))
        self.robot.switch_tool_sprung_finger_tip_left()
        distance2 = max(0, block_length_max - np.linalg.norm(
            self.tower.initial_pos[:2] - self.robot.get_world_position()[:2]))
        self.robot.switch_tool_two_fingers()
        distance3 = max(0, block_length_max - np.linalg.norm(
            self.tower.initial_pos[:2] - self.robot.get_world_position()[:2]))
        distance = max(distance1, distance2, distance3)
        self.move_along_own_axis('x', distance, speed=speed)

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

    def get_state_advanced(self, new_block, force=0, block_displacement=0, mode=0, poses=None):
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
            if poses is None:
                poses = self.tower.get_poses_cv()
            block_positions = self.tower.get_positions_from_poses(poses)
            block_orientations = self.tower.get_orientations_from_poses(poses)


            # filter out poses calculated using only one tag
            reduced_poses = dict()
            for i in poses:
                if poses[i]['tags_detected'][0] == 2:
                    reduced_poses[i] = poses[i]
            if not reduced_poses:
                reduced_poses = poses
            reduced_positions = self.tower.get_positions_from_poses(reduced_poses)
            reduced_orientations = self.tower.get_orientations_from_poses(reduced_poses)


            log.debug(f"After get positions")

            # calculate tilt
            log.debug(f"Before get tilt")
            # tilt_2ax = self.tower.get_tilt_2ax(block_positions, block_orientations, self.current_block_id)
            tilt_2ax = self.tower.get_tilt_2ax(reduced_positions, reduced_orientations, self.current_block_id)
            log.debug(f"After get tilt")
            if new_block:
                self.initial_tilt_2ax = tilt_2ax
                self.previous_step_displacement = np.array([0, 0])
                self.previous_step_z_rot = 0
            last_tilt_2ax = tilt_2ax - self.initial_tilt_2ax

            # get last z_rotation
            log.debug(f"Before get z rotaiton")
            # z_rotation_last = self.tower.get_last_z_rotation(self.current_block_id, block_orientations)
            z_rotation_last = self.tower.get_last_z_rotation(self.current_block_id, reduced_orientations)
            log.debug(f"After get z rotaiton")

            # get current round displacement
            # current_round_displacement = self.tower.get_last_displacement_2ax(self.current_block_id, block_positions, block_orientations)
            current_round_displacement = self.tower.get_last_displacement_2ax(self.current_block_id, reduced_positions, reduced_orientations)
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
            layer_configuration = self.calculate_layer_configuration(block_positions, block_orientations)

            # forma a state
            state = np.array([force, block_displacement, total_block_distance, current_step_tower_displacement[0],
                             current_step_tower_displacement[1], current_round_displacement[0],
                             current_round_displacement[1], last_tilt_2ax[0], last_tilt_2ax[1], current_step_z_rot,
                             z_rotation_last, side, block_height, self.current_lvl_pos, layer_configuration])
            log.debug(f"Get state#2")

            if self.plot_state_data:
                self.update_env_state_plot(state)

            return state, poses

    def get_state(self, new_block, force=0, block_displacement=0, mode=0, poses=None):
        """
        mode=0: return gym compatible state
        mode=1: return state as State class
        """


        if mode == 0:

            log.debug(f"Get state#1")

            # get block positions
            log.debug(f"Before get positions")

            if poses is None:
                poses = self.tower.get_poses_cv()
            block_positions = self.tower.get_positions_from_poses(poses)
            block_orientations = self.tower.get_orientations_from_poses(poses)

            # filter out poses calculated using only one tag
            reduced_poses = dict()
            for i in poses:
                if poses[i]['tags_detected'][0] == 2 and i in self.blocks_for_features:
                    reduced_poses[i] = poses[i]
            if not reduced_poses:
                reduced_poses = poses

            if not new_block:
                self.blocks_for_features = self.blocks_for_features.intersection(set(reduced_poses.keys()))

            reduced_poses = {i: reduced_poses[i] for i in reduced_poses if i in self.blocks_for_features}

            reduced_positions = self.tower.get_positions_from_poses(reduced_poses)
            reduced_orientations = self.tower.get_orientations_from_poses(reduced_poses)

            ###############333 DEBUUUGUUGGUUGGUGGGGG
            detected_blocks = []
            for i in reduced_poses:
                detected_blocks.append(i)
            detected_blocks.sort()
            print(f"Detected blocks: {detected_blocks}")
            print(f"Num of detected blocks: {len(detected_blocks)}")
            displacements = {}
            for i in detected_blocks:
                displacements[i] = np.linalg.norm(self.positions_from_last_step[i] - block_positions[i])
            displacements = {k: v for k, v in sorted(displacements.items(), key=lambda item: item[1])}
            print(f"Detected blocks displacements: {displacements}")

            ####################################33


            log.debug(f"After get positions")

            # calculate tilt
            log.debug(f"Before get tilt")

            log.debug(f"After get tilt")
            if new_block:
                self.previous_step_displacement = 0
                self.previous_step_max_displacement = 0

                # save detected blocks that will be used for features calculating
                self.blocks_for_features = set(reduced_poses.keys())

                for i in reduced_positions:
                    self.positions_from_last_step[i] = reduced_positions[i]

            # get last z_rotation
            log.debug(f"Before get z rotaiton")
            log.debug(f"After get z rotaiton")

            # get current round displacement
            # current_round_displacement = self.tower.get_last_displacement_2ax(self.current_block_id, block_positions, block_orientations)
            current_round_displacement = self.tower.get_last_displacement_2ax(self.current_block_id, reduced_positions, reduced_orientations)
            log.debug(f"After get displacement")

            # get max displacement
            current_round_block_max_displacement = self.tower.get_last_max_displacement_2ax(self.current_block_id, reduced_positions,
                                                                              reduced_orientations)

            # get last step tower displacement and z rotation
            current_step_tower_displacement = np.linalg.norm(self.get_last_step_avg_disp(reduced_positions, reduced_orientations))
            current_step_max_displacement = max(0, np.linalg.norm(self.get_last_step_max_disp(reduced_positions, reduced_orientations))- 1)

            # reset previous displacement and z rotation
            self.previous_step_displacement = current_round_displacement
            self.previous_step_max_displacement = current_round_block_max_displacement
            for i in reduced_positions:
                self.positions_from_last_step[i] = reduced_positions[i]

            total_block_distance = self.total_distance

            # get block height
            # block_height = block_positions[self.current_block_id][2] / one_millimeter
            block_height = self.robot.get_world_position()[2]

            # get layer configuration
            # 0) □□□
            # 1) □x□
            # 2) □□x
            # 3) x□□
            # 4) x□x
            layer_configuration = self.calculate_layer_configuration(block_positions, block_orientations)

            # forma a state
            state = np.array([force,
                              block_displacement,
                              total_block_distance,
                              current_step_tower_displacement,
                              np.linalg.norm(current_round_displacement),
                              current_step_max_displacement,
                              block_height])
                              # self.current_lvl_pos,
                              # layer_configuration])
                # ,
                #               self.current_block_id])
            log.debug(f"Get state#2")

            return state, poses

    def calculate_layer_configuration(self, positions, orientations):
        # get layer configuration
        # 0) □□□
        # 1) □x□
        # 2) □□x
        # 3) x□□
        # 4) x□x
        log.debug(f"Before get layers state")
        layers = self.tower.get_layers_state(positions, orientations, real_tag_pos)
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

        return layer_configuration

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
        self.robot.set_world_pos_orientation_mov(intermediate_pos, zwischenablage_place_quat, speed=10, pos_flags='(7, 0)')

        # move to the end target
        self.robot.set_world_pose_mov(zwischenablage_place_pose, degrees=True, speed=10, pos_flags='(7, 0)')

        # release block
        self.robot.open_wide()

        # move upwards
        self.move_upwards(150)

    def take_from_zwischenablage(self):
        # move to intermediate pos
        intermediate_pos = zwischenablage_take_pos + zwischenablage_take_quat.rotate(x_unit_vector) * 50
        self.robot.set_world_pos_orientation(intermediate_pos, zwischenablage_take_quat, speed=15)

        # move to the final target
        self.robot.set_world_pose(zwischenablage_take_pose, degrees=True, speed=15)

        # grip
        self.robot.grip()

        # move upwards
        current_pos = self.robot.get_world_position()
        waiting_pos = current_pos
        waiting_pos[2] = stopover_height
        self.robot.set_world_pos_orientation(waiting_pos, Quaternion(axis=z_unit_vector, degrees=45), speed=15)

    def calculate_gripper_orientation(self, block_orientation):
        dot_x = np.dot(block_orientation.rotate(x_unit_vector), x_unit_vector)
        dot_y = np.dot(block_orientation.rotate(x_unit_vector), y_unit_vector)

        orientation1 = block_orientation * Quaternion(axis=z_unit_vector, degrees=90)
        orientation2 = block_orientation * Quaternion(axis=z_unit_vector, degrees=-90)
        if abs(dot_x) > 0.5:
            if np.dot(orientation1.rotate(x_unit_vector), y_unit_vector) < 0:
                return orientation1
            if np.dot(orientation2.rotate(x_unit_vector), y_unit_vector) < 0:
                return orientation2
        if abs(dot_y) >= 0.5:
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
        self.robot.set_world_pos_orientation_mov(intermediate_pos, gripper_orientation, pos_flags='(7, 0)', speed=10)

        # go to the last stopover near the tower
        self.robot.set_world_pos_orientation_mov(last_stopover, gripper_orientation, pos_flags='(7, 0)', speed=10)

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

        poses = self.tower.get_poses_cv()
        positions = self.tower.get_positions_from_poses(poses)
        orientations = self.tower.get_orientations_from_poses(poses)

        # calculate corrections for the orientations
        self.tower.calculate_orientation_corrections(orientations)

    def check_blocks(self):
        self.go_home()

        for j in range(30):
            block_id, lvl, pos = self.move_to_random_block()
            for i in range(30):
                force, displacement, poses = self.push()
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
        print(f"Loose blocks: {loose_blocks}")
        for block_id, lvl, pos in loose_blocks:
            poses = self.tower.get_poses_cv()
            self.move_to_block_id_push(block_id, poses)
            total_distance = 0
            for i in range(30):
                force, displacement, poses = self.push()
                total_distance += displacement
                print(f"Step #i: Force: {force}, displacement: {displacement}")
                if force > self.get_force_threshold(lvl):
                    break
                if i > 10:
                    self.push_through(30 - total_distance)
                    input('Confirm extraction:')
                    # self.extract_and_place_on_top(block_id)
                    break


    def check_certain_blocks(self):
        self.go_home()

        loose_blocks = [(39, 14, 2)]

        for block_id, lvl, pos in loose_blocks:
            poses = self.tower.get_poses_cv()
            self.move_to_block_id_push(block_id, poses)
            total_distance = 0
            for i in range(20):
                force, displacement, poses = self.push()
                total_distance += displacement
                print(f"Step #i: Force: {force}, displacement: {displacement}")
                if force > self.get_force_threshold(lvl):
                    break
                if i > 10:
                    self.push_through(20 - total_distance)
                    # input('Confirm extraction:')

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
        for i in layers:
            if i > 2:
                id0 = layers[i][0]
                id1 = layers[i][1]
                id2 = layers[i][2]
                if id0 is not None and id1 is not None and id2 is not None:
                    height0 = self.block_sizes[id0]['height']
                    height1 = self.block_sizes[id1]['height']
                    height2 = self.block_sizes[id2]['height']

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

    # returns observation, reward, done, info
    def step(self, action):
        step_start_time = time.time()
        normalize = self.normalize
        log.debug(f"Jenga step#1")
        log.debug(f"Jenga step#2")

        if action == 0:  # switch block
            log.debug(f"Jenga step#3")
            # reset state variables
            self.total_distance = 0
            self.steps_pushed = 0
            log.debug(f"Before move to random block")
            block_id, lvl, pos = self.move_to_random_block()
            log.debug(f"After move to random block")

            info = {'extracted_blocks': self.extracted_blocks}
            log.debug(f"Before get state!")

            poses = self.tower.get_poses_cv()
            positions = self.tower.get_positions_from_poses(poses)
            orientations = self.tower.get_orientations_from_poses(poses)

            # reset last reference positions and orientations
            self.tower.last_ref_positions = copy.deepcopy(positions)
            self.tower.last_ref_orientations = copy.deepcopy(orientations)

            # reinitialize blocks that can be using for feature calculation
            self.blocks_for_features = set([i for i in range(g_blocks_max)])

            state, poses = self.get_state(new_block=True, poses=poses)

            log.debug(f"After get state!")

            # normalize state
            if normalize:
                state = self.normalize_state(state)

            # check whether the game is over or not
            positions = self.tower.get_positions_from_poses(poses)
            tower_toppled = self.tower.toppled(positions, self.current_block_id)
            done = block_id is None or tower_toppled

            print(f"Block_id: {block_id}")
            print(f"Tower toppled: {tower_toppled}")

            if tower_toppled:
                reward = tower_toppled_reward
            else:
                reward = 0

            # save the last state and reward
            self.last_state = state
            self.last_reward = reward

            # set reference force
            self.set_reference_force()

            if self.normalize:
                state_for_plotting = state * state_space_stds_real_robot + state_space_means_real_robot

            if self.plot_state_data:
                self.update_env_state_plot(state_for_plotting, reward)

            log.debug(f"Jenga step#4")
            return state, reward, done, info

        if action == 1:  # extract and switch block
            log.debug(f"Jenga step#5")
            # extract and move to the next block
            if self.steps_pushed == self.individual_pushing_distance:
                log.debug(f"Jenga step#6")
                # reset state variables
                self.total_distance = 0
                self.steps_pushed = 0

                # push through remaining distance
                self.push_through(self.max_pushing_distance - self.individual_pushing_distance)

                log.debug(f"Before pull_and_place")
                self.extract_and_place_on_top(self.current_block_id)
                self.go_home()
                # input(f"Confirm extracting: ")
                log.debug(f"After pull_and_place")



                info = {'extracted_blocks': self.extracted_blocks}

                poses = self.tower.get_poses_cv()
                positions = self.tower.get_positions_from_poses(poses)
                orientations = self.tower.get_orientations_from_poses(poses)

                # reinitialize blocks that can be using for feature calculation
                self.blocks_for_features = set([i for i in range(g_blocks_max)])

                # reset last reference positions and orientations
                self.tower.last_ref_positions = copy.deepcopy(positions)
                self.tower.last_ref_orientations = copy.deepcopy(orientations)

                log.debug(f"Before  get state")
                state, poses = self.get_state(new_block=True, poses=poses)
                log.debug(f"After  get state")

                # normalize state
                if normalize:
                    state = self.normalize_state(state)

                log.debug(f"Before move_to_random_block")
                block_id, lvl, block_pos = self.move_to_random_block()
                log.debug(f"After move_to_random_block")
                self.extracted_blocks += 1

                # check whether the game is over or not
                positions = self.tower.get_positions_from_poses(poses)
                tower_toppled = self.tower.toppled(positions, self.current_block_id)
                done = block_id is None or tower_toppled

                print(f"Block_id: {block_id}")
                print(f"Tower toppled: {tower_toppled}")

                if tower_toppled:
                    reward = tower_toppled_reward
                else:
                    reward = reward_extract

                # save the last state and reward
                self.last_state = state
                self.last_reward = reward

                if self.normalize:
                    state_for_plotting = state * state_space_stds_real_robot + state_space_means_real_robot

                if self.plot_state_data:
                    self.update_env_state_plot(state_for_plotting, reward)

                # set reference force
                self.set_reference_force()

                log.debug(f"Jenga step#7")
                return state, reward, done, info

            else:  # push
                log.debug(f"Jenga step#8")

                start = time.time()
                force, displacement, poses = self.push()
                elapsed = time.time() - start
                print(f'Push time: {elapsed*1000:.2f}ms')

                self.set_reference_force()

                log.debug(f"Pushed!")
                self.steps_pushed += 1
                log.debug(f"Before get state")
                start = time.time()
                state, poses = self.get_state(new_block=False, force=force, block_displacement=displacement, poses=poses)
                elapsed = time.time() - start
                print(f'Get state time: {elapsed*1000:.2f}ms')
                log.debug(f"After get state")

                # check whether the game is over or not
                start = time.time()
                positions = self.tower.get_positions_from_poses(poses)

                tower_toppled = self.tower.toppled(positions, self.current_block_id)
                elapsed = time.time() - start
                print(f'Toppled time: {elapsed * 1000:.2f}ms')
                done = tower_toppled

                print(f"Tower toppled: {tower_toppled}")

                if tower_toppled:
                    reward = tower_toppled_reward
                else:
                    reward = self.compute_reward(state, normalize)


                info = {'extracted_blocks': self.extracted_blocks}

                # normalize state
                if normalize:
                    state = self.normalize_state(state)

                # save the last state and reward
                self.last_state = state
                self.last_reward = reward

                log.debug(f"Jenga step#9")

                step_elapsed = time.time() - step_start_time
                print(f"Step elapsed total: {step_elapsed:.3f}s")

                if self.normalize:
                    state_for_plotting = state * state_space_stds_real_robot + state_space_means_real_robot

                if self.plot_state_data:
                    self.update_env_state_plot(state_for_plotting, reward)

                return state, reward, done, info

    def reset(self):
        # go home
        self.go_home()

        # wait until tower reset
        input('Confirm tower reset: ')  # user should enter some input into terminal to confirm tower reset

        # reset global variables
        self.initialize_global_variables()

        # move to some block
        # self.move_to_random_block()

        poses = self.tower.get_poses_cv()
        positions = self.tower.get_positions_from_poses(poses)
        orientations = self.tower.get_orientations_from_poses(poses)

        # reset last reference positions and orientations
        self.tower.last_ref_positions = copy.deepcopy(positions)
        self.tower.last_ref_orientations = copy.deepcopy(orientations)
        for i in positions:
            self.positions_from_last_step[i] = positions[i]

        # get initial state
        state, poses = self.get_state(True, poses=poses)

        print(f"State: {state}")



        if self.normalize:
            state_for_plotting = state * state_space_stds_real_robot + state_space_means_real_robot

        # update state plot
        self.update_env_state_plot(state_for_plotting, 0)

        return state

    def close(self):
        self.go_home()
        input('Confirm closing: ')

    def normalize_state(self, state):
        return (state - state_space_means_real_robot) / state_space_stds_real_robot

    def normalize_reward(self, reward):
        return (reward - reward_mean_real_robot) / reward_std_real_robot

    def compute_reward_advanced(self, state, normalize):
        block_displacement = state[1]
        tilt = state[7:9]
        current_step_tower_displacement = state[3:5]
        current_step_tower_z_rotation = state[9]

        tower_displacement_1ax = np.linalg.norm(current_step_tower_displacement)
        tower_tilt_1ax = np.linalg.norm(tilt)
        z_rot = abs(current_step_tower_z_rotation)
        coefficients = np.array([1, -1.5, -1, -2])

        reward = sum(coefficients * np.array([block_displacement, tower_displacement_1ax, tower_tilt_1ax, z_rot]))

        # normalize reward
        if normalize:
            reward = self.normalize_reward(reward)

        return reward

    def compute_reward(self, state, normalize):
        block_displacement = state[1]
        current_step_tower_displacement = state[3]
        current_step_max_displacement = state[5]

        tower_displacement_1ax = np.linalg.norm(current_step_tower_displacement)
        coefficients = np.array([1, -6, -3])

        reward = sum(coefficients * np.array([block_displacement, tower_displacement_1ax, current_step_max_displacement]))

        # normalize reward
        if normalize:
            reward = self.normalize_reward(reward)

        return reward

    def get_last_step_avg_disp(self, positions, orientations):
        return self.tower._get_displacement_2ax(self.current_block_id, self.positions_from_last_step, positions, orientations)

    def get_last_step_max_disp(self, positions, orientations):
        return self.tower._get_last_max_displacement_2ax(self.current_block_id, self.positions_from_last_step, positions, orientations)

if __name__ == "__main__":
    jenga = jenga_env(True)

    poses = jenga.tower.get_poses_cv()
    # positions = jenga.tower.get_positions_from_poses(poses)
    # layers = jenga.tower.get_layers(positions)

    # jenga.get_blocks_to_check_ml()
    #
    # exit()
    # jenga.check_extractable_blocks()
    # jenga.extract_certain_blocks()

    # jenga.move_to_block_push(4, 1, poses)

    # put on zwischenablage

    jenga.take_from_zwischenablage()
    jenga.place_on_top(39)

    # jenga.take_from_zwischenablage()

    exit()

    poses_list = []

    initial_poses = jenga.tower.get_poses_cv()

    for i in range(10):
        start = time.time()
        poses = jenga.tower.get_poses_cv()
        elapsed = time.time() - start
        print(f'Elapsed time: {elapsed*1000:.2f}ms')
        poses_list.append(poses)

    deviations = {}
    for id in initial_poses:
        deviations[id] = []
        for poses in poses_list:
            if id in poses:
                deviation = np.linalg.norm(poses[id]['pos'] - initial_poses[id]['pos'])
                deviations[id].append(deviation)
        deviations[id] = np.mean(deviations[id])

    for i in deviations:
        print(f"Mean deviation of block #{i} = {deviations[i]}")


    # print(jenga.find_extractable_blocks())
    #
    # jenga.check_extractable_blocks()

