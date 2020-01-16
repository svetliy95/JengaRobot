from constants import *
from math import sqrt
from pyquaternion import Quaternion
from tower import Tower
import time

class Extractor:

    size = 0.005 * scaler
    width = 0.030 * scaler
    finger_length = 0.020 * scaler
    finger_mass = 0.0125 * scaler ** 3
    finger_kp = finger_mass * 10000
    finger_damping = 2 * finger_mass * sqrt(finger_kp / finger_mass)  # critical damping
    HOME_POS = np.array([-0.3, 0, 2 * size / scaler]) * scaler

    base_mass = 0.5 * scaler ** 3
    total_mass = base_mass + 2 * finger_mass
    base_kp = total_mass * 1000
    base_damping = 2 * total_mass * sqrt(base_kp / total_mass)  # critical damping

    finger1_pos = 0
    finger2_pos = 0

    distance_between_fingers_close = block_width_mean * 0.8
    distance_between_fingers_open = block_width_mean * 1.6

    translation_to_move_slowly = np.zeros(3)



    def __init__(self, sim, tower: Tower):
        self.sim = sim
        self.x = self.HOME_POS[0]
        self.y = self.HOME_POS[1]
        self.z = self.HOME_POS[2]
        self.q = Quaternion(axis=[0, 0, 1], degrees=180)
        self.tower = tower
        self.t = 0
        self.speed = 0.01  # in mm/timestep

    def set_position(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

    def get_position(self):
        return np.array([self.x, self.y, self.z])

    def set_orientation(self, q: Quaternion):
        self.q = q

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

        # update position
        self.sim.data.ctrl[6] = self.x - self.HOME_POS[0]
        self.sim.data.ctrl[7] = self.y - self.HOME_POS[1]
        self.sim.data.ctrl[8] = self.z - self.HOME_POS[2]

        # self.sim.data.ctrl[6] = self.x
        # self.sim.data.ctrl[7] = self.y
        # self.sim.data.ctrl[8] = self.z

        # updata orientation
        yaw_pitch_roll = self.q.yaw_pitch_roll
        self.sim.data.ctrl[9] = yaw_pitch_roll[2]
        self.sim.data.ctrl[10] = yaw_pitch_roll[1]
        self.sim.data.ctrl[11] = yaw_pitch_roll[0]

        # update finger positions
        self.sim.data.ctrl[12] = self.finger1_pos
        self.sim.data.ctrl[13] = self.finger2_pos

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

    def open(self):
        self.set_finger_distance(self.distance_between_fingers_open)

    def close(self):
        self.set_finger_distance(self.distance_between_fingers_close)


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
        z = pos[2]
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
        print(f"Start quarter: {start_quarter}.")
        print(f"End quarter: {end_quarter}.")
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

        return stopovers

    def go_home(self):
        self.set_position(self.HOME_POS)

    def put_on_top(self):
        heighest_block_z = self.tower.get_position(self.tower.get_highest_block_num())[2]
        height = heighest_block_z + 1

        stopover = self.get_position()
        stopover[2] = height

        self.translation_to_move_slowly = stopover - self.get_position()

        while np.any(self.translation_to_move_slowly):
            self._sleep_simtime(0.2)

        target = np.array([0, 0, height])

        self.translation_to_move_slowly = target - self.get_position()

        while np.any(self.translation_to_move_slowly):
            self._sleep_simtime(0.2)

        self.open()


    def pull(self, block_id):
        block_quat = Quaternion(self.tower.get_orientation(block_id)) * Quaternion(axis=[0, 0, 1], degrees=180)
        block_pos = np.array(self.tower.get_position(block_id))
        block_x_face_normal_vector = np.array(block_quat.rotate(x_unit_vector))
        target = np.array(block_pos + (block_length_mean + (
                    Extractor.finger_length * 2 + Extractor.size)) * block_x_face_normal_vector)
        self.translation_to_move_slowly = target - block_pos

        while np.any(self.translation_to_move_slowly):
            self._sleep_simtime(0.2)

    def translate(self, v_translation):
        assert isinstance(v_translation, np.ndarray), "Translation vector must be of type np.ndarray"
        assert v_translation.size == 3, 'Incorrect translation vector dimensionality.'
        self.x += v_translation[0]
        self.y += v_translation[1]
        self.z += v_translation[2]

    # The path is calculated according to the quarters in which extractor and the target are located
    def move_to_block(self, num):
        block_quat = Quaternion(self.tower.get_orientation(num)) * Quaternion(axis=[0, 0, 1], degrees=180)
        block_pos = np.array(self.tower.get_position(num))
        block_x_face_normal_vector = np.array(block_quat.rotate(x_unit_vector))

        target = np.array(block_pos + (block_length_mean/2 - block_width_mean + (Extractor.finger_length * 2 + Extractor.size)) * block_x_face_normal_vector)
        intermediate_target = target + (block_length_mean/2) * block_x_face_normal_vector

        stopovers = self._get_stopovers(self._get_quarter(self.get_position()),
                                        self._get_quarter(target),
                                        target[2])

        for stopover in stopovers:
            self.set_position(stopover)
            self._sleep_simtime(0.2)

        self.set_orientation(block_quat)
        self._sleep_simtime(0.2)
        self.set_position(intermediate_target)
        self._sleep_simtime(0.2)

        self.set_position(target)
        self._sleep_simtime(0.2)

        self.close()
        self._sleep_simtime(0.2)
        self.pull(num)
        self._sleep_simtime(0.3)
        self.put_on_top()





        # # move backwards to avoid collisions
        # self.move_along_own_axis('x', block_length_mean)
        # self._sleep_simtime(0.3)
        #
        # # move along its own y-axis towards the intermediate target
        # self.move_along_own_axis_towards_point('y', intermediate_target)
        # self._sleep_simtime(0.3)
        #
        # # rotate
        # self.set_orientation(block_quat)
        # self._sleep_simtime(0.3)
        #
        # # translate towards the intermediate target
        # self.set_position(intermediate_target)
        # self._sleep_simtime(0.3)
        #
        # # translate towards the target
        # self.set_position(pos=target)

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
                                <geom type="box" size="{Extractor.finger_length} {Extractor.size} {Extractor.size}" mass="{Extractor.finger_mass}"/>
                            </body>
                            <body name="finger2" pos="{-Extractor.finger_length - Extractor.size} {Extractor.width - Extractor.size} 0">
                                <joint name="finger2_joint" pos="0 0 0" type="slide" axis="0 1 0" damping="{Extractor.finger_damping}"/>
                                <geom type="box" size="{Extractor.finger_length} {Extractor.size} {Extractor.size}" mass="{Extractor.finger_mass}"/>
                            </body>                            
                            <geom type="box" size="0.4 0.4 0.4" mass="0.1" rgba="1 0 0 1" pos="0 0 0"/>
                        </body>'''
        return f'''<body name="extractor" pos="{Extractor.HOME_POS[0]} {Extractor.HOME_POS[1]} {Extractor.HOME_POS[2]}" euler="0 0 0">
                    <joint name="extractor_slide_x" type="slide" pos="{Extractor.size + 2*Extractor.finger_length} 0 0" axis="1 0 0" damping="{Extractor.base_damping}"/>
                    <joint name="extractor_slide_y" type="slide" pos="{Extractor.size + 2*Extractor.finger_length} 0 0" axis="0 1 0" damping="{Extractor.base_damping}"/>
                    <joint name="extractor_slide_z" type="slide" pos="{Extractor.size + 2*Extractor.finger_length} 0 0" axis="0 0 1" damping="{Extractor.base_damping}"/>
                    <joint name="extractor_hinge_x" type="hinge" axis="1 0 0" damping="{Extractor.base_damping}" pos ="{Extractor.size + 2*Extractor.finger_length} 0 0"/>
                    <joint name="extractor_hinge_y" type="hinge" axis="0 1 0" damping="{Extractor.base_damping}" pos ="{Extractor.size + 2*Extractor.finger_length} 0 0"/>
                    <joint name="extractor_hinge_z" type="hinge" axis="0 0 1" damping="{Extractor.base_damping}" pos ="{Extractor.size + 2*Extractor.finger_length} 0 0"/>
                    <geom type="box" pos="{Extractor.size + 2*Extractor.finger_length} 0 0" size="{Extractor.size} {Extractor.width} {Extractor.size}" mass="{Extractor.base_mass}"/>
                    <body name="finger1" pos="{Extractor.finger_length} {-Extractor.width + Extractor.size} 0">
                        <joint name="finger1_joint" pos="0 0 0" type="slide" axis="0 1 0" damping="{Extractor.finger_damping}"/>
                        <geom type="box" size="{Extractor.finger_length} {Extractor.size} {Extractor.size}" mass="{Extractor.finger_mass}"/>
                    </body>
                    <body name="finger2" pos="{Extractor.finger_length} {Extractor.width - Extractor.size} 0">
                        <joint name="finger2_joint" pos="0 0 0" type="slide" axis="0 1 0" damping="{Extractor.finger_damping}"/>
                        <geom type="box" size="{Extractor.finger_length} {Extractor.size} {Extractor.size}" mass="{Extractor.finger_mass}"/>
                    </body>
                </body>'''
