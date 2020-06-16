import numpy as np
from pyquaternion import Quaternion

# simulation
g_timestep = 0.005

# tower
g_blocks_num = 30
scaler = 50
one_millimeter = 0.001 * scaler

# coordinate axes parameter
coordinate_axes_pos_x = -0.3 * scaler
coordinate_axes_pos_y = -0.3 * scaler
coordinate_axes_pos_z = 0
coordinate_axes_pos = np.array([coordinate_axes_pos_x, coordinate_axes_pos_y, coordinate_axes_pos_z])
coordinate_axes_width = 0.005 * scaler
coordinate_axes_height = 0.025 * scaler

# coordinate frame tag
coordinate_frame_tag_size = np.array([0.06*scaler, 0.06*scaler, 0.02])
coordinate_frame_tag_pos = np.array([-0.1*scaler, -0.1*scaler, 0])
coordinate_frame_tag_id = 255

# empirical data
block_length_mean = 0.0750283018867925 * scaler
block_length_sigma = 0.000278476182167942 * scaler
block_length_min = 0.0744 * scaler
block_length_max = 0.0757 * scaler

block_width_mean = 0.0249274074074074 * scaler
block_width_sigma = 0.000245396753720476 * scaler
block_width_min = 0.02405 * scaler
block_width_max = 0.02535 * scaler

block_height_mean = 0.0148412037037037 * scaler
block_height_sigma = 0.000175380812576479 * scaler
block_height_min = 0.0145 * scaler
block_height_max = 0.0152 * scaler

block_mass_mean = 0.0192679811320755 * scaler**3
block_mass_sigma = 0.001406754796448 * scaler**3
block_mass_min = 0.015876 * scaler**3
block_mass_max = 0.022995 * scaler**3

# vectors
x_unit_vector = np.array([1, 0, 0])
y_unit_vector = np.array([0, 1, 0])
z_unit_vector = np.array([0, 0, 1])

# sensing
timesteps_transient = 75
displacement_threshold = 0.3
pusher_spring_length = 0.1 * scaler

# camera parameters
fovy = 45

# tower parameters
same_height_threshold = block_height_mean/3
origin = np.array([0, 0, 0])
toppled_distance = 1*block_length_mean
toppled_block_threshold = 5
block_to_far_threshold = np.linalg.norm(block_height_mean * z_unit_vector +
                         (block_length_mean - block_width_mean/2) * x_unit_vector +
                         (block_width_mean) * y_unit_vector) * 2

# zwischenablage
zwischanablage_pos = np.array([-0.2, -0.3, 0.1]) * scaler
zwischenablage_quat = Quaternion(axis=z_unit_vector, degrees=45) * Quaternion(axis=y_unit_vector, degrees=45) * Quaternion(axis=x_unit_vector, degrees=45)
zwischenablage_quat_elem = zwischenablage_quat.q
zwischanablage_base_size = np.array([block_width_mean / scaler, block_length_mean/2 / scaler, 0.001]) * scaler
zwischanablage_bottom_wall_size = np.array([0.001, block_length_mean/2 / scaler, block_height_mean*0.8/scaler]) * scaler
zwischanablage_side_wall_size = np.array([block_width_mean / scaler, 0.001, block_height_mean*0.2/scaler]) * scaler

# timeouts in seconds
timeout_push = 5
timeout_pull = 20
timeout_move = 10

# rewards
reward_extract = 4

# rl normalization coefficients
state_space_means = np.array([31093.59760076597, 0.7089517916864881, 14.293707360944792, -0.012084712192316339, -0.00490070252700048, -0.0345395773469817, 0.08062475922521721, -0.049612524438130065, -0.02525760238493288, 0.005152267459163039, 0.12637937423004747, 0.322, 54.051269341582])
state_space_stds = np.array([99413.88253555144, 0.42956519713406727, 13.34898031972997, 0.09598252282445836, 0.10812004345160166, 0.3045609955626232, 0.4481523369012735, 0.10575237652977863, 0.07723860768596473, 0.06639173492656888, 0.5704620884914603, 0.4672429774753174, 39.63450085951062])
action_mean = 0.972
action_std = 0.16497272501841023
reward_mean = 0.4863470043130892
reward_std = 0.5801684522930456
