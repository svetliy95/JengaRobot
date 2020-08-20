import numpy as np
from math import cos, sin, pi, radians


def pose2matrix_XYZ(pose):
    X = pose[0]
    Y = pose[1]
    Z = pose[2]
    A = pose[3]
    B = pose[4]
    C = pose[5]

    T = np.array([[1, 0, 0, X],
                  [0, 1, 0, Y],
                  [0, 0, 1, Z],
                  [0, 0, 0, 1]])
    Rz = np.array([[cos(C), -sin(C), 0, 0],
                   [sin(C), cos(C), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[cos(B), 0, sin(B), 0],
                   [0, 1, 0, 0],
                   [-sin(B), 0, cos(B), 0],
                   [0, 0, 0, 1]])

    Rx = np.array([[1, 0, 0, 0],
                   [0, cos(A), -sin(A), 0],
                   [0, sin(A), cos(A), 0],
                   [0, 0, 0, 1]])

    return T @ Rx @ Ry @ Rz


def pose2matrix_ZYX(pose):
    X = pose[0]
    Y = pose[1]
    Z = pose[2]
    A = pose[3]
    B = pose[4]
    C = pose[5]

    T = np.array([[1, 0, 0, X],
                  [0, 1, 0, Y],
                  [0, 0, 1, Z],
                  [0, 0, 0, 1]])
    Rz = np.array([[cos(A), -sin(A), 0, 0],
                   [sin(A), cos(A), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[cos(B), 0, sin(B), 0],
                   [0, 1, 0, 0],
                   [-sin(B), 0, cos(B), 0],
                   [0, 0, 0, 1]])

    Rx = np.array([[1, 0, 0, 0],
                   [0, cos(C), -sin(C), 0],
                   [0, sin(C), cos(C), 0],
                   [0, 0, 0, 1]])

    return T @ Rz @ Ry @ Rx


# Ausgabe von Pose in X, Y, Z, Rz, Ry', Rx''
def matrix2pose_ZYX(a_matrix):
    l_norm = np.sqrt(np.square(a_matrix[0, 0]) + np.square(a_matrix[1, 0]))

    if l_norm > 0.00001:
        l_sa = a_matrix[1, 0] / l_norm
        l_ca = a_matrix[0, 0] / l_norm
        l_WA = np.arctan2(l_sa, l_ca)
        l_WB = np.arctan2(-a_matrix[2, 0], l_ca * a_matrix[0, 0] + l_sa * a_matrix[1, 0])
        l_WC = np.arctan2(l_sa * a_matrix[0, 2] - l_ca * a_matrix[1, 2], -l_sa * a_matrix[0, 1] + l_ca * a_matrix[1, 1])
    else:
        l_WA = 0
        l_WB = np.arctan2(-a_matrix[2, 0], l_norm)

        if l_WB > 0:
            l_WC = np.arctan2(a_matrix[0, 1], a_matrix[1, 1])
        else:
            l_WC = -np.arctan2(a_matrix[0, 1], a_matrix[1, 1])

    l_pose = np.array([a_matrix[0, 3], a_matrix[1, 3], a_matrix[2, 3], l_WA, l_WB, l_WC])
    return l_pose

def matrix2pose_XYZ(a_matrix):
    l_norm = np.sqrt(np.square(a_matrix[0, 0]) + np.square(a_matrix[1, 0]))

    if l_norm > 0.00001:
        l_sa = a_matrix[1, 0] / l_norm
        l_ca = a_matrix[0, 0] / l_norm
        l_WA = np.arctan2(l_sa, l_ca)
        l_WB = np.arctan2(-a_matrix[2, 0], l_ca * a_matrix[0, 0] + l_sa * a_matrix[1, 0])
        l_WC = np.arctan2(l_sa * a_matrix[0, 2] - l_ca * a_matrix[1, 2], -l_sa * a_matrix[0, 1] + l_ca * a_matrix[1, 1])
    else:
        l_WA = 0
        l_WB = np.arctan2(-a_matrix[2, 0], l_norm)

        if l_WB > 0:
            l_WC = np.arctan2(a_matrix[0, 1], a_matrix[1, 1])
        else:
            l_WC = -np.arctan2(a_matrix[0, 1], a_matrix[1, 1])

    l_pose = np.array([a_matrix[0, 3], a_matrix[1, 3], a_matrix[2, 3], l_WA, l_WB, l_WC])
    return l_pose

# accepts angle in degrees
# returns homogeneous rotation matrix
def get_Rx_h(theta, units="degrees"):
    assert units in ["degrees", "radians"]
    if units is "degrees":
        theta = radians(theta)

    return np.array([[1, 0, 0, 0],
                    [0, cos(theta), -sin(theta), 0],
                    [0, sin(theta), cos(theta), 0],
                    [0, 0, 0, 1]])


# accepts angle in degrees
# returns homogeneous rotation matrix
def get_Ry_h(theta, units="degrees"):
    assert units in ["degrees", "radians"]
    if units is "degrees":
        theta = radians(theta)

    return np.array([[cos(theta), 0, sin(theta), 0],
                    [0, 1, 0, 0],
                    [-sin(theta), 0, cos(theta), 0],
                    [0, 0, 0, 1]])


# accepts angle in degrees
# returns homogeneous rotation matrix
def get_Rz_h(theta, units="degrees"):
    assert units in ["degrees", "radians"]
    if units is "degrees":
        theta = radians(theta)

    return np.array([[cos(theta), -sin(theta), 0, 0],
                    [sin(theta), cos(theta), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])


# accepts angle in degrees
# returns rotation matrix
def get_Rx(theta, units="degrees"):
    assert units in ["degrees", "radians"]
    if units is "degrees":
        theta = radians(theta)

    return np.array([[1, 0, 0],
                    [0, cos(theta), -sin(theta)],
                    [0, sin(theta), cos(theta)]])


# accepts angle in degrees
# returns rotation matrix
def get_Ry(theta, units="degrees"):
    assert units in ["degrees", "radians"]
    if units is "degrees":
        theta = radians(theta)

    return np.array([[cos(theta), 0, sin(theta)],
                    [0, 1, 0],
                    [-sin(theta), 0, cos(theta)]])


# accepts angle in degrees
# returns rotation matrix
def get_Rz(theta, units="degrees"):
    assert units in ["degrees", "radians"]
    if units is "degrees":
        theta = radians(theta)

    return np.array([[cos(theta), -sin(theta), 0],
                    [sin(theta), cos(theta), 0],
                    [0, 0, 1]])


def getRotationMatrix_90_x():
    return np.array([[1, 0, 0, 0],
                     [0, 0, -1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])


def getRotationMatrix_90_y():
    return np.array([[-1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])


def getRotationMatrix_90_z():
    return np.array([[0, -1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def getRotationMatrix_180_x():
    return np.array([[1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, -1, 0],
                     [0, 0, 0, 1]])


def getDrone2CameraTransformMatrix():
    return np.array([[-1, 0, 0, 0],
                     [0, 0, -1, 140],
                     [0, -1, 0, 280],
                     [0, 0, 0, 1]])


# test
if __name__ == '__main__':
    pose = [2661, 1941, 1307, -144 * pi / 180, -7 * pi / 180, 82 * pi / 180]
    pose2 = [0, 0, 0, 180 * pi / 180, 0 * pi / 180, -90 * pi / 180]
    pose3 = [1256, 3843, 1606, 81 * pi / 180, 0.71 * pi / 180, -178 * pi / 180]
    pose4 = [1250, 3800, 1600, 81 * pi / 180, 0.71 * pi / 180, -178 * pi / 180]
    pose5 = [1250, 3800, 1600, 90 * pi / 180, 0 * pi / 180, -180 * pi / 180]

    print(pose5)
    matrix = pose2matrix_ZYX(pose5)
    print(matrix)
    # r_90_matrix = np.matrix([[1, 0, 0, 0],
    #                              [0, 0, 1, 0],
    #                              [0, -1, 0, 0],
    #                              [0, 0, 0, 1]])
    matrix2 = matrix * getDrone2CameraTransformMatrix()
    print(matrix2)

    pose5 = matrix2pose_ZYX(matrix2)
    print(pose5)
