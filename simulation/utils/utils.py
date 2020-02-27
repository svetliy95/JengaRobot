import numpy as np
import math
from pyquaternion import Quaternion
from constants import *


def point_projection_on_line(line_point1, line_point2, point):
    ap = point - line_point1
    ab = line_point2 - line_point1
    result = line_point1 + np.dot(ap, ab) / np.dot(ab, ab) * ab
    return result


def get_direction_towards_origin_along_vector(vec, p,  origin=np.array([0, 0, 0])):
    # normalize vector
    vec = vec / np.linalg.norm(vec)
    first_point = p
    second_point = p + vec
    origin_projection = point_projection_on_line(first_point, second_point, origin)
    direction = origin_projection - p

    # normalize
    direction = direction / np.linalg.norm(direction)

    return direction

def normalize_angle(angle, units):
    assert units in ['degrees', 'radians']

    if units == 'degrees':
        angle = math.radians(angle)

    if angle >= 0 and (angle // math.pi) % 2 == 0:
        angle = angle % math.pi

    if angle >= 0 and (angle // math.pi) % 2 == 1:
        angle = -(math.pi - (angle % math.pi))

    if angle < 0 and (-angle // math.pi) % 2 == 0:
        angle = -(-angle % math.pi)

    if angle < 0 and (-angle // math.pi) % 2 == 1:
        angle = (math.pi - (-angle % math.pi))

    if units == "degrees":
        angle = math.degrees(angle)

    return angle

def restrict_euler_angle_interval(yaw_pitch_roll, units):
    assert units in ['degrees', 'radians']
    yaw = yaw_pitch_roll[0]
    pitch = yaw_pitch_roll[1]
    roll = yaw_pitch_roll[2]
    if units == 'degrees':
        yaw = math.radians(yaw)
        pitch = math.radians(pitch)
        roll = math.radians(roll)

    if yaw >= 0 and (yaw // math.pi) % 2 == 0:
        yaw = yaw % math.pi

    if yaw >= 0 and (yaw // math.pi) % 2 == 1:
        yaw = -(math.pi - (yaw % math.pi))

    if yaw < 0 and (-yaw // math.pi) % 2 == 0:
        yaw = -(-yaw % math.pi)

    if yaw < 0 and (-yaw // math.pi) % 2 == 1:
        yaw = (math.pi - (-yaw % math.pi))

    return yaw

# This function returns intermediate rotations that pass the origin instead of passing -180/180 degrees
def get_intermediate_rotations(q1: Quaternion, q2: Quaternion, steps):
    yaw1 = q1.yaw_pitch_roll[0]
    yaw2 = q2.yaw_pitch_roll[0]
    intermediate_quaternions = []

    if yaw1 < 0 and yaw2 >= 0:  # if only one of the angles is negative
        if abs(yaw1 + math.pi) + abs(math.pi - yaw2) < abs(yaw1) + abs(yaw2):
            difference = abs(yaw1 - yaw2)
            # create two intermediate rotations
            intermediate_q1 = q1 * Quaternion(axis=[0, 0, 1], radians=difference/3)
            intermediate_q2 = q1 * Quaternion(axis=[0, 0, 1], radians=2 * (difference/3))

            intermediate_q = Quaternion.intermediates(q1, intermediate_q1, steps//3)
            for i in range(steps//3):
                intermediate_quaternions.append(next(intermediate_q))
            intermediate_q = Quaternion.intermediates(intermediate_q1, intermediate_q2, steps//3)
            for i in range(steps//3):
                intermediate_quaternions.append(next(intermediate_q))
            intermediate_q = Quaternion.intermediates(intermediate_q2, q2, steps // 3)
            for i in range(steps//3):
                intermediate_quaternions.append(next(intermediate_q))
            intermediate_quaternions.append(q2)
    elif yaw1 >= 0 and yaw2 < 0:
        if abs(math.pi - yaw1) + abs(yaw2 + math.pi) < abs(yaw1) + abs(yaw2):
            difference = abs(yaw1 - yaw2)
            # create two intermediate rotations
            intermediate_q1 = q2 * Quaternion(axis=[0, 0, 1], radians=2 * (difference / 3))
            intermediate_q2 = q2 * Quaternion(axis=[0, 0, 1], radians=difference / 3)

            intermediate_q = Quaternion.intermediates(q1, intermediate_q1, steps // 3)
            for i in range(steps // 3):
                intermediate_quaternions.append(next(intermediate_q))
            intermediate_q = Quaternion.intermediates(intermediate_q1, intermediate_q2, steps // 3)
            for i in range(steps // 3):
                intermediate_quaternions.append(next(intermediate_q))
            intermediate_q = Quaternion.intermediates(intermediate_q2, q2, steps // 3)
            for i in range(steps // 3):
                intermediate_quaternions.append(next(intermediate_q))
            intermediate_quaternions.append(q2)

    if not any(intermediate_quaternions):
        intermediate_q = Quaternion.intermediates(q1, q2, steps, True)
        for i in range(steps):
            intermediate_quaternions.append(next(intermediate_q))

    return intermediate_quaternions

def angle_between_vectors(a, b):
    return math.acos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_angle_between_quaternions(q1, q2):
    q1 = Quaternion(q1)
    q2 = Quaternion(q2)
    v1 = q1.rotate(x_unit_vector)
    v2 = q2.rotate(x_unit_vector)
    x_error = angle_between_vectors(v1, v2)
    v1 = q1.rotate(y_unit_vector)
    v2 = q2.rotate(y_unit_vector)
    y_error = angle_between_vectors(v1, v2)
    v1 = q1.rotate(z_unit_vector)
    v2 = q2.rotate(z_unit_vector)
    z_error = angle_between_vectors(v1, v2)

    return max(x_error, y_error, z_error)

