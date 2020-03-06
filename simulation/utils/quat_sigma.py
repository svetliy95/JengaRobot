from pyquaternion import Quaternion
import numpy as np
from numpy.random import normal
from utils import get_angle_between_quaternions as angle
import math

q = Quaternion(np.random.random(4))

angles = []
distorted_quats = []
sigma_degrees = 4/3
Sigma = 0.01 * np.ones(4)
for i in range(1000):
    q_distorted = q * \
        Quaternion(axis=[1, 0, 0], degrees=normal(0, sigma_degrees)) * \
        Quaternion(axis=[0, 1, 0], degrees=normal(0, sigma_degrees)) * \
        Quaternion(axis=[0, 0, 1], degrees=normal(0, sigma_degrees))
    distorted_quats.append(q_distorted)
    angles.append(angle(q, q_distorted))

print(f"Original: {q.q}")
angles = np.array(angles)
print(f"Mean: {math.degrees(np.mean(angles))}")
print(f"Distorted quats: {distorted_quats}")



