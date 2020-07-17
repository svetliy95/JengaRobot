import socket
import numpy as np
from constants import right_robot_ip, right_robot_port, buffer_size
import time
import random

p1 = np.array([ 319.45, -292.58,  443.23, -178.43,   -3.35,   -1.9 ])
p1 = np.array([ 319.45, -292.58,  443.23, -180,   0,   0 ])
p2 = np.array([ 319.42,   46.17,  443.23, -180,   0,   0 ])
p3 = np.array([ 618.26,   48.24,  443.2 , -180,   0,   0])
p4 = np.array([ 534.16, -298.34,  443.2 , -180,   0,   0])
p5 = np.array([ 484.69,  -46.91,  650.54, -180,   0,   0 ])



class Robot:

    moveToPosID = '1001'
    getPosID = '2002'
    ackFlag = 777

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def connect(self):
        self.socket.connect((self.ip, self.port))

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

    def get_pos(self):
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

    def recv(self):
        data = self.socket.recv(1024)
        data = data.decode('ascii')
        data = data[:-2]
        return data

if __name__ == "__main__":
    r = Robot(right_robot_ip, right_robot_port)
    r.connect()

    poss = [p1, p2, p3, p4, p5]

    for i in range(10):
        pos = random.choice(poss)
        r.mvs(pos)

