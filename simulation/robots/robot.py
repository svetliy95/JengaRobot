import socket
import numpy as np
from constants import *
import time
import random
import matplotlib; matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import animation
style.use('fivethirtyeight')
from utils.utils import plane_normal_from_points, define_axis, calculate_rotation, x_unit_vector, y_unit_vector, z_unit_vector, right_robot_home_position_world
from cv.transformations import matrix2pose_XYZ, pose2matrix_XYZ, pose2matrix_ZYX, matrix2pose_ZYX, get_Rz_h, get_Ry_h, get_Rx_h
import math
from pyquaternion import Quaternion
import json
from robots.gripper import Gripper
from utils.utils import euler2quat, quat2euler

p1 = np.array([ 319.45, -292.58,  443.23, -178.43,   -3.35,   -1.9 ])
p1 = np.array([ 319.45, -292.58,  443.23, -180,   0,   0 ])
p2 = np.array([ 319.42,   46.17,  443.23, -180,   0,   0 ])
p3 = np.array([ 618.26,   48.24,  443.2 , -180,   0,   0])
p4 = np.array([ 534.16, -298.34,  443.2 , -180,   0,   0])
p5 = np.array([ 484.69,  -46.91,  650.54, -180,   0,   0 ])

class CoordinateSystem:
    def __init__(self, vx, vy, vz, origin):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.origin = origin

    @classmethod
    def from_three_points(cls, origin, x_ax, y_ax):
        vx, vy, vz = CoordinateSystem.calculate_base_vectors(origin, x_ax, y_ax)
        return cls(vx, vy, vz, origin)

    # for ratation
    def get_transformation_mtx(self):
        B = np.array([self.vx, self.vy, self.vz, self.origin]).transpose()
        B = np.concatenate((B, np.reshape([0, 0, 0, 1], (1, 4))), axis=0)
        A = np.eye(4)
        T = np.linalg.inv(A) @ B

        return T

    # calculates transformation from robot coord system to
    @staticmethod
    def calculate_base_vectors(origin, x_ax, y_ax):
        vz = plane_normal_from_points(origin, x_ax, y_ax)
        vx = define_axis(origin, x_ax, vz)
        vy = np.cross(vz, vx)

        return [vx, vy, vz]

class Robot:

    mvsFuncID = '1001'
    getPosID = '2002'
    getForceID = '3003'
    switchToolID = '4004'
    movFuncID = '5005'
    ackFlag = 777

    def __init__(self, ip, port, coord_system, gripper: Gripper):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.force_buffer = []
        self.sensors_data_queue_maxsize = 250
        self.sensor_data_queue = []
        self.coord_system = coord_system
        self.gripper = gripper

    def connect(self):
        self.socket.connect((self.ip, self.port))
        print(f"Connected!")

    def disconnect(self):
        self.socket.close()
        print(f"Disconnected!")

    def send(self, msg_str):
        # prepare message
        msg_str = msg_str.encode('ascii')

        # send message
        self.socket.send(msg_str)

        # wait until done
        ans = self.recv()

        # check ack
        if int(ans) != Robot.ackFlag:
            print(f"Error!")

    def mvs(self, pos, pos_flags, speed):
        assert len(pos) == 6, "Pos should be a 6d vector"

        # choose function
        self.send(Robot.mvsFuncID)

        # prepare message
        msg_str = f'({pos[0]}, {pos[1]}, {pos[2]}, {pos[3]}, {pos[4]}, {pos[5]}){pos_flags}'

        # send pose
        self.send(msg_str)

        # send speed
        self.send(str(speed))

    def mov(self, pos, pos_flags, speed):
        assert len(pos) == 6, "Pos should be a 6d vector"

        # choose function
        self.send(Robot.movFuncID)

        # prepare message
        msg_str = f'({pos[0]}, {pos[1]}, {pos[2]}, {pos[3]}, {pos[4]}, {pos[5]}){pos_flags}'

        # send pos
        self.send(msg_str)

        # send speed
        self.send(str(speed))

    def get_pose(self):
        # choose function
        self.socket.send(Robot.getPosID.encode('ascii'))

        # receive pos as a string
        pos_str = self.recv()

        # remove brackets
        pos_str = pos_str[1:-1]

        # remove FLG1 and FLG2 (see robot manual)
        pos = pos_str.split(')')[0]

        # split at commas
        pos = pos.split(',')

        # convert to array
        pos = np.array(list(map(float, pos)))
        return pos

    def get_world_pose(self, degrees):
        pose = self.get_pose()
        pose[3:] = pose[:-4:-1]

        pose[3:] = np.array([math.radians(pose[3]), math.radians(pose[4]), math.radians(pose[5])])
        mtx = pose2matrix_ZYX(pose)

        mtx = mtx @ self.tool_transformation_matrix()

        mtx = np.linalg.inv(self.coord_system.get_transformation_mtx()) @ mtx

        pose = matrix2pose_ZYX(mtx)

        pose[3:] = pose[:-4:-1]

        if degrees:
            pose[3:] = np.array([math.degrees(pose[3]), math.degrees(pose[4]), math.degrees(pose[5])])

        return pose

    def get_world_position(self):
        pose = self.get_world_pose(degrees=False)
        pos = pose[:3]
        return pos

    def get_world_orientation(self):
        pose = self.get_world_pose(degrees=False)
        quat = euler2quat(pose[3:], degrees=False)

        return quat

    def get_pose_in_robot(self):
        return self.get_pose()

    def set_world_pos(self, pos, speed=0):
        current_pose = self.get_world_pose(degrees=False)
        current_pose[:3] = pos
        self.set_world_pose(current_pose, degrees=False, speed=speed)

    def set_world_pos_orientation(self, pos, quat: Quaternion, pos_flag='', speed=0):
        euler = quat2euler(quat)
        pose = np.concatenate((pos, euler))
        self.set_world_pose(pose, degrees=False, pos_flags=pos_flag, speed=speed)

    def set_world_pos_orientation_mov(self, pos, quat: Quaternion, pos_flags='', speed=0):
        euler = quat2euler(quat)
        pose = np.concatenate((pos, euler))
        self.set_world_pose_mov(pose, degrees=False, pos_flags=pos_flags, speed=speed)

    def set_world_quat(self, quat, pos_flags='', speed=0):
        current_pose = self.get_world_pose(degrees=False)
        euler = quat2euler(quat)
        current_pose[3:] = euler
        self.set_world_pose(current_pose, degrees=False, pos_flags=pos_flags, speed=speed)

    def set_world_pose(self, pose, degrees, pos_flags='', speed=0):
        pose = pose.astype(np.float)
        # pose = np.array([403.76 ,  70.565, 300.548, 179.789,  -0.086,  43.922])
        pose[3:] = pose[:-4:-1]
        if degrees:
            pose[3:] = np.array(list(map(math.radians, pose[3:])))

        mtx = pose2matrix_ZYX(pose)



        mtx = self.coord_system.get_transformation_mtx() @ mtx

        # rotate tool coordinate system
        mtx = mtx @ self.tool_transformation_matrix()

        pose = matrix2pose_ZYX(mtx)

        pose[3:] = pose[:-4:-1]



        # express euler angle in degrees
        pose[3:] = np.array(list(map(math.degrees, pose[3:])))

        self.mvs(pose, pos_flags, speed)

    def set_world_pose_mov(self, pose, degrees, pos_flags='', speed=0):
        pose = pose.astype(np.float)
        # pose = np.array([403.76 ,  70.565, 300.548, 179.789,  -0.086,  43.922])
        pose[3:] = pose[:-4:-1]
        if degrees:
            pose[3:] = np.array(list(map(math.radians, pose[3:])))

        mtx = pose2matrix_ZYX(pose)



        mtx = self.coord_system.get_transformation_mtx() @ mtx

        # rotate tool coordinate system
        mtx = mtx @ self.tool_transformation_matrix()

        pose = matrix2pose_ZYX(mtx)

        pose[3:] = pose[:-4:-1]



        # express euler angle in degrees
        pose[3:] = np.array(list(map(math.degrees, pose[3:])))

        self.mov(pose, pos_flags, speed)

    def get_force(self):
        # choose function
        self.socket.send(Robot.getForceID.encode('ascii'))

        # receive force as a string
        force_str = self.recv()

        # remove brackets
        force_str = force_str[1:-1]

        # remove FLG1 and FLG2 (see robot manual)
        force = force_str.split(')')[0]

        # split at commas
        force = force.split(',')

        # convert to array
        force = np.array(list(map(float, force)))
        return force

    def plot_forces(self, axis):
        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure()
        ax = plt.axes(xlim=(0, self.sensors_data_queue_maxsize), ylim=(-30, 30))
        line, = ax.plot([], [], lw=2)
        if axis == 'x':
            data_index = 0
        if axis == 'y':
            data_index = 1
        if axis == 'z':
            data_index = 2
        if axis == 'Mx':
            data_index = 3
        if axis == 'My':
            data_index = 4
        if axis == 'Mz':
            data_index = 5


        # initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            return line,

        # animation function.  This is called sequentially
        def animate(i):
            current_sensor_value = self.get_force()
            if len(self.sensor_data_queue) >= self.sensors_data_queue_maxsize:
                self.sensor_data_queue.pop(0)
            self.sensor_data_queue.append(current_sensor_value)
            x = np.linspace(0, len(self.sensor_data_queue), len(self.sensor_data_queue))
            if axis is None:
                y = [data for data in self.sensor_data_queue]
            else:
                y = [data[data_index] for data in self.sensor_data_queue]
            minimum = min(y)
            maximum = max(y)
            mean = np.mean(y)
            ax.set_ylim((minimum - abs(mean-minimum)*0.1, maximum + abs(maximum - mean)*0.1))
            line.set_data(x, y)
            return line,

        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=200, interval=30, blit=False)

        plt.show()

    def recv(self):
        data = self.socket.recv(buffer_size)
        data = data.decode('ascii')
        data = data[:-2]
        return data

    def tool_transformation_matrix(self):
        mtx = get_Rx_h(180, 'degrees')# @ get_Rz_h(180, 'degrees')
        return mtx

    def write_forces(self):
        forces1 = []
        forces2 = []
        for i in range(1000):
            forces1.append(list(r1.get_force()))
            forces2.append(list(r2.get_force()))

        with open("jitter_data/forces_r1_b5_d3_g1.json", 'w') as f:
            json.dump(forces1, f)

        with open("jitter_data/forces_r2_b1_d3_g1.json", 'w') as f:
            json.dump(forces2, f)

    def switch_tool(self, id):
        # choose function
        self.socket.send(Robot.switchToolID.encode('ascii'))

        # check ack flag
        if int(self.recv()) != Robot.ackFlag:
            print(f"Error!")

        # prepare message
        msg_str = f'{id}'.encode('ascii')

        # send message
        self.socket.send(msg_str)

        # wait until done
        ans = self.recv()

        # check ack
        if int(ans) != Robot.ackFlag:
            print(f"Error!")

    def switch_tool_taster(self):
        self.switch_tool(1)

    def switch_tool_two_fingers(self):
        self.switch_tool(2)

    def switch_tool_sprung_finger_tip_right(self):
        self.switch_tool(3)

    def switch_tool_sprung_finger_tip_left(self):
        self.switch_tool(4)

    def switch_tool_center_of_area(self):
        self.switch_tool(5)

    def grip(self):
        self.gripper.grip()
        self.gripper.wait_until_gripping_done()

    def release(self):
        self.gripper.release()
        self.gripper.wait_until_release()

    def open_wide(self):
        self.gripper.move_to_pos(100, 200)
        self.gripper.wait_until_positioning_done()

if __name__ == "__main__":

    scaler = 0.01

    # x-ax [ 631.31   65.82    1.38 -179.76    5.56   17.85]
    # y-ax [ 397.7  -279.84    1.09 -179.75    5.57   17.86]
    # origin [ 404.85   70.08    0.55 -177.81    3.5    -2.07]
    x_ax = np.array([631.31, 65.82, 1.38])
    y_ax = np.array([397.7, -279.84, 1.09])
    origin = np.array([404.85, 70.08, 0.55])

    # initialize coordinate system
    x_ax = np.array([404.36, -91.24, 0.36])
    y_ax = np.array([331.34, 307.78, 1.09])
    origin = np.array([565.7, 65.05, 0.56])
    coord_system = CoordinateSystem.from_three_points(origin, x_ax, y_ax)

    # initialize grippers
    g1 = Gripper(right_gripper_ip)
    g2 = Gripper(left_gripper_ip)

    # initialize robots
    r1 = Robot(right_robot_ip, right_robot_port, coord_system, g1)
    r2 = Robot(left_robot_ip, left_robot_port, coord_system, g2)
    r1.connect()
    # r2.connect()

    pos = r1.get_world_position()
    quat = r1.get_world_orientation()
    pos = pos - quat.rotate(x_unit_vector)

    # r1.set_world_pos(pos)
    r1.plot_forces('x')

    r1.switch_tool_two_fingers()
    np.set_printoptions(precision=3, suppress=True)
    print(f"Robot pose: {r1.get_pose()}")
    print(repr(r1.get_world_pose(degrees=True)))
    print(repr(r1.get_world_position()))
    print(repr(r1.get_world_orientation()))
    print(list(map(math.degrees, quat2euler(r1.get_world_orientation()))))
    print(euler2quat(r1.get_world_pose(degrees=False)[3:], degrees=False))


    exit()

    # r1.open_wide()
    r1.set_world_pose_mov(right_robot_home_position_world, degrees=True, pos_flags='(7, 0)', detour=0)

    # r1.set_world_pose(np.array([+495.02,+22.77,+300.50,-179.81,-0.04,+177.22]), degrees=True, pos_flags='(7, 0)', detour=0)

    exit()

    import cv2
    from cv.calculate_zwischenablage_pose import get_zwischenablage_pose
    import glob

    fnames = glob.glob('../cv/pictures/IMG_*.jpg')

    images = []

    for f in fnames:
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        images.append(im)

    pos, quat, t_mtx, pose = get_zwischenablage_pose(images, phone_cam_mtx, phone_cam_dist)
    print(f"Pos: {pos}")
    print(f"Quat: {quat}")
    print(f"Transformation matrix: {t_mtx}")
    print(f"Pose: {repr(pose)}")

    # exit()

    # r1.switch_tool_sprung_finger_tip()
    # r1.set_world_pose_mov(pose + np.array([0, 0, 50, 0, 0, 0]), degrees=True)
    r1.switch_tool_two_fingers()
    r1.set_world_pose(pose + np.array([0, 0, 50, 0, 0, 0]), degrees=True)

    exit()


    r1.disconnect()
    # r2.disconnect()



