import numpy as np
import math
from pyquaternion import Quaternion
from constants import *
import traceback
from scipy import stats


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
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    cos = bound(-1, 1, cos)  # bound the value berween -1 and 1, because because of floating point arithmetic the dot
                             # product can be grater than 1
    return math.acos(cos)


def get_angle_between_quaternions(q1, q2):
    return max(get_angle_between_quaternions_3ax(q1, q2))

def get_angle_between_quaternions_3ax(q1, q2):
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

    return np.array([x_error, y_error, z_error])

def orth_proj(v, u):
    return (np.dot(u, v)/np.linalg.norm(v)**2) * v

def proj_on_plane(n, v):
    return v - orth_proj(n, v)

def remove_outliers(data, treshold=3):
    data = np.array(data)
    z_score = np.array(stats.zscore(data))
    idx1 = treshold > z_score
    idx2 = -treshold < z_score
    data = data[np.logical_and(idx1, idx2)]

    return data


def translation_along_axis_towards_point(src_point, dst_point, axis, quat):
    translation_line = quat.rotate(axis) / np.linalg.norm(axis)  # rotate and normalize
    projection = point_projection_on_line(src_point, src_point + translation_line, dst_point)

    # calculate direction
    direction1 = translation_line
    direction2 = -translation_line
    dist1 = np.linalg.norm(src_point + direction1 - dst_point)
    dist2 = np.linalg.norm(src_point + direction2 - dst_point)

    if dist1 < dist2:
        return np.linalg.norm(projection - src_point)
    else:
        return -np.linalg.norm(projection - src_point)

def quat_canonical_form(q):
    if q[0] < 0:
        return -q
    else:
        return q

# computes diffs in mtx elements in percents
def mtx_diff(mtx1, mtx2):
    return np.divide((mtx1 - mtx2), mtx1) * 100

def bound(low, high, value):
    return max(low, min(high, value))

def plane_normal_from_points(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    normal = np.cross(v1, v2)

    # normalize
    normal = normal / np.linalg.norm(normal)

    return normal

def define_axis(p1, p2, plane_normal):
    v = p2 - p1
    axis_vector = proj_on_plane(plane_normal, v)

    # normalize
    axis_vector = axis_vector / np.linalg.norm(axis_vector)

    return axis_vector

def calculate_rotation(v1, v2):
    cross_product = np.cross(v1, v2)
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    w = math.sqrt(v1_norm**2 * v2_norm**2) + np.dot(v1, v2)

    quat = Quaternion([w, cross_product[0], cross_product[1], cross_product[2]]).normalised

    return quat