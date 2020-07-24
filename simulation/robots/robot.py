import socket
import numpy as np
from constants import right_robot_ip, right_robot_port, left_robot_port, left_robot_ip, buffer_size
import time
import random
import matplotlib; matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import animation
style.use('fivethirtyeight')
from utils.utils import plane_normal_from_points, define_axis, calculate_rotation, x_unit_vector, y_unit_vector, z_unit_vector
from cv.transformations import matrix2pose_XYZ, pose2matrix_XYZ, get_Rz_h, get_Ry_h, get_Rx_h
import math

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

    # for ratation
    def get_transformation_mtx(self):
        B = np.array([self.vx, self.vy, self.vz, self.origin]).transpose()
        B = np.concatenate((B, np.reshape([0, 0, 0, 1], (1, 4))), axis=0)
        A = np.eye(4)
        T = np.linalg.inv(A) @ B

        return T

class Robot:

    moveToPosID = '1001'
    getPosID = '2002'
    getForceID = '3003'
    ackFlag = 777

    def __init__(self, ip, port, coord_system):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.force_buffer = []
        self.sensors_data_queue_maxsize = 250
        self.sensor_data_queue = []

    def connect(self):
        self.socket.connect((self.ip, self.port))
        print(f"Connected!")

    def mvs(self, pos):
        assert len(pos) == 6, "Pos should be a 6d vector"

        # choose function
        self.socket.send(Robot.moveToPosID.encode('ascii'))

        # check ack flag
        if int(self.recv()) != Robot.ackFlag:
            print(f"Error!")

        # prepare message
        msg_str = f'({pos[0]}, {pos[1]}, {pos[2]}, {pos[3]}, {pos[4]}, {pos[5]})'.encode('ascii')

        # send message
        self.socket.send(msg_str)

        # wait until done
        ans = self.recv()

        # check ack
        if int(ans) != Robot.ackFlag:
            print(f"Error!")

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

    def get_pose_in_world(self, degrees=False):
        pose = self.get_pose()
        pose[3:] = np.array([math.radians(pose[3]), math.radians(pose[4]), math.radians(pose[5])])
        print(f"Pose vorher: {pose}")
        mtx = pose2matrix_XYZ(pose)

        # rotate tool coordinate system
        mtx = mtx @ self.tool_transformation_matrix()

        mtx = np.linalg.inv(coord_system.get_transformation_mtx()) @ mtx
        pose = matrix2pose_XYZ(mtx)

        if degrees:
            pose[3:] = np.array([math.degrees(pose[3]), math.degrees(pose[4]), math.degrees(pose[5])])

        return pose

    def get_pose_in_robot(self):
        return self.get_pose()

    def set_world_pose(self, pose, deegrees=False):
        pose = pose.astype(np.float)
        if deegrees:
            pose[3:] = np.array(list(map(math.radians, pose[3:])))

        print(f"Pose in radians: {pose}")

        mtx = pose2matrix_XYZ(pose)

        # rotate tool coordinate system
        mtx = mtx @ self.tool_transformation_matrix()

        mtx = coord_system.get_transformation_mtx() @ mtx
        pose = matrix2pose_XYZ(mtx)

        pose[3:] = np.array(list(map(math.degrees, pose[3:])))

        print(f"Robot pose2: {pose}")
        self.mvs(pose)

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
        mtx = get_Rx_h(180, 'degrees')
        return mtx

    # calculates transformation from robot coord system to
    @staticmethod
    def calculate_base_vectors(origin, x_ax, y_ax):
        vz = plane_normal_from_points(origin, x_ax, y_ax)
        vx = define_axis(origin, x_ax, vz)
        vy = np.cross(vz, vx)

        return [vx, vy, vz]


if __name__ == "__main__":

    scaler = 0.01

    # x-ax [ 631.31   65.82    1.38 -179.76    5.56   17.85]
    # y-ax [ 397.7  -279.84    1.09 -179.75    5.57   17.86]
    # origin [ 404.85   70.08    0.55 -177.81    3.5    -2.07]
    x_ax = np.array([631.31, 65.82, 1.38])
    y_ax = np.array([397.7, -279.84, 1.09])
    origin = np.array([404.85, 70.08, 0.55])
    vx, vy, vz = Robot.calculate_base_vectors(origin, x_ax, y_ax)
    coord_system = CoordinateSystem(vx, vy, vz, origin)
    coord_system = CoordinateSystem(x_unit_vector, y_unit_vector, z_unit_vector, origin)


    r1 = Robot(right_robot_ip, right_robot_port, coord_system)
    r2 = Robot(left_robot_ip, left_robot_port, coord_system)
    r1.connect()
    r2.connect()

    # pose = r1.get_pose()
    # pose[3:] = np.array([0, 0, 0])
    # mtx = pose2matrix_XYZ(pose)
    # mtx = mtx @ get_Rx_h(180)
    # pose = matrix2pose_XYZ(mtx)
    # pose[3:] = np.array([math.degrees(pose[3]), math.degrees(pose[4]), math.degrees(pose[5])])
    # r1.mvs(pose)
    np.set_printoptions(precision=3, suppress=True)
    pose = np.array([0, 0, 300, 0, 0, 45])

    r1.set_world_pose(pose, deegrees=True)

