from robots.robot import Robot, CoordinateSystem
import numpy as np
from constants import *
from random import random
import time
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib; matplotlib.use("TkAgg")
import csv
from matplotlib import rc

sns.set_theme(style="whitegrid")
rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=14)
rc('legend', fontsize=13)
rc('text.latex', preamble=r'\usepackage{cmbright}')

# measures and averages forces to get the reference force
reference_force = 0
def set_reference_force():
    global reference_force
    reference_force = get_averaged_force(10)

def get_averaged_force(n=5):
    forces = []
    for i in range(n):
        force = r.get_force()[0]
        forces.append(force)

    return np.mean(forces)

def calculate_displacement_from_force(measured_force):
    robot_displacement = -measured_force/right_robot_spring_constant

    return robot_displacement

def move_along_own_axis(axis, distance, speed=0):  # distance can be negative
    assert axis == 'x' or axis == 'y' or axis == 'z', "Wrong axis!"

    if axis == 'x':
        unit_vector = x_unit_vector
    if axis == 'y':
        unit_vector = y_unit_vector
    if axis == 'z':
        unit_vector = z_unit_vector

    gripper_quat = r.get_world_orientation()
    gripper_pos = r.get_world_position()

    translation_direction = np.array(gripper_quat.rotate(unit_vector))
    end_point = np.array([gripper_pos[0], gripper_pos[1], gripper_pos[2]]) + translation_direction * distance
    r.set_world_pos(end_point, speed=speed)





if __name__ == "__main__":
    # # initialize coordinate system
    # x_ax = np.array([404.36, -91.24, 0.36])
    # y_ax = np.array([331.34, 307.78, 1.09])
    # origin = np.array([565.7, 65.05, 0.56])
    # coord_system = CoordinateSystem.from_three_points(origin, x_ax, y_ax)
    # r = Robot(right_robot_ip, right_robot_port, coord_system, None)
    # r.connect()
    #
    #
    # for i in range(100):
    #     displacement = 7*random() + 0.1
    #     set_reference_force()
    #     move_along_own_axis('x', -displacement)
    #     time.sleep(read_force_wait*2)
    #     force = get_averaged_force(5) - reference_force
    #     estimated_displacement = calculate_displacement_from_force(force)
    #     print(f"True vs estimated displacement: {displacement:.2f}, {estimated_displacement:.2f}, {estimated_displacement-displacement:.2f}")
    #     move_along_own_axis('x', displacement)
    #     time.sleep(read_force_wait*2)
    #     with open('displacements_left.csv', 'a+') as f:
    #         f.write(f"{displacement}, {estimated_displacement}\n")

    ###### evaluate ####
    true = []
    estimated = []
    diff = []
    with open('displacements_left.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            true.append(float(row[0]))
            estimated.append(float(row[1]))
            diff.append(float(row[1]) - float(row[0]))

    mean_diff = np.mean(diff)
    print(f"Mean difference: {mean_diff}")
    print(f"Std difference: {np.std(diff)}")

    new_diff = []
    for i in diff:
        new_diff.append(abs(i - mean_diff))
        print(i - mean_diff)

    mean_diff = np.mean(new_diff)
    print(f"Mean difference: {mean_diff}")
    print(f"Std difference: {np.std(new_diff)}")

    fig_dims = (5.6, 1.5)
    fig, ax = plt.subplots(figsize=fig_dims)

    # sns.displot(data=new_diff, bins=25)
    ax = sns.boxplot(data=new_diff, orient='h')
    plt.tight_layout()
    ax.set(ylabel='', xlabel='Absoluter Fehler in mm')
    plt.show()


