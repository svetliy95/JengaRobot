from constants import *
from math import sqrt
from pyquaternion import Quaternion
from tower import Tower

class Extractor:

    size = 0.005 * scaler
    width = 0.030 * scaler
    finger_length = 0.020 * scaler
    finger_mass = 0.0125 * scaler ** 3
    finger_kp = finger_mass * 1000
    finger_damping = 2 * finger_mass * sqrt(finger_kp / finger_mass)  # critical damping
    STARTING_POS = np.array([-0.1, 0, 2 * size / scaler]) * scaler

    base_mass = 0.125 * scaler ** 3
    total_mass = base_mass + 2 * finger_mass
    base_kp = total_mass * 1000
    base_damping = 2 * total_mass * sqrt(base_kp / total_mass)  # critical damping

    finger1_pos = 0
    finger2_pos = 0

    distance_between_fingers_close = block_width_mean * 0.8
    distance_between_fingers_open = block_width_mean * 1.6



    def __init__(self, sim, tower: Tower):
        self.sim = sim
        self.x = Extractor.STARTING_POS[0]
        self.y = Extractor.STARTING_POS[1]
        self.z = Extractor.STARTING_POS[2]
        self.q = Quaternion([1, 0, 0, 0])
        self.tower = tower

    def set_position(self, pos):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_orientation(self, q: Quaternion):
        self.q = q

    def update_positions(self):
        # update position
        self.sim.data.ctrl[6] = self.x - self.STARTING_POS[0]
        self.sim.data.ctrl[7] = self.y - self.STARTING_POS[1]
        self.sim.data.ctrl[8] = self.z - self.STARTING_POS[2]

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


    def move_to_block(self, num):
        block_quat = self.tower.get_orientation(num)
        block_pos = self.tower.get_position(num)
        block_x_face_normal_vector = block_quat.rotate(-x_unit_vector)

        target = block_pos + (block_length_mean/2 - block_width_mean) * block_x_face_normal_vector
        intermediate_target = ...



    @staticmethod
    def generate_xml():
        return f'''<body name="extractor" pos="{Extractor.STARTING_POS[0]} {Extractor.STARTING_POS[1]} {Extractor.STARTING_POS[2]}" euler="0 0 180">
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
