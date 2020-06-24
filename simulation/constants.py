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
toppled_block_threshold = 3
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
timeout_step = 20

# rewards
reward_extract = 4
tower_toppled_reward = -100

# rl normalization coefficients
state_space_means = np.array([31093.59760076597, 0.7089517916864881, 14.293707360944792, -0.012084712192316339, -0.00490070252700048, -0.0345395773469817, 0.08062475922521721, -0.049612524438130065, -0.02525760238493288, 0.005152267459163039, 0.12637937423004747, 0.322, 54.051269341582])
state_space_stds = np.array([99413.88253555144, 0.42956519713406727, 13.34898031972997, 0.09598252282445836, 0.10812004345160166, 0.3045609955626232, 0.4481523369012735, 0.10575237652977863, 0.07723860768596473, 0.06639173492656888, 0.5704620884914603, 0.4672429774753174, 39.63450085951062])
action_mean = 0.972
action_std = 0.16497272501841023
reward_mean = 0.4863470043130892
reward_std = 0.5801684522930456

# normalization coefficients based on 5 games
state_space_means = np.array([4229.508076832736, 0.6922730870034125, 7.736966353950943, -0.003858395892212402, -0.003427758135832582, 0.013664164624334639, 0.029472820731486517, -0.035724613574329954, -0.023702386526712804, -0.00040170530023534486, -0.008084520644138997, 0.5064275037369208, 111.2938368862308])
state_space_stds = np.array([26675.70656671306, 0.4675007370441492, 7.547234435571376, 0.07682875478361244, 0.0758656643758738, 0.18417095717895468, 0.17879750884265974, 0.08709192558984649, 0.07096177534308634, 0.023181354669204318, 0.13166999428775308, 0.49995868548882083, 65.89176301267419])
reward_mean = 0.667859775832977
reward_std = 0.7651965181928698
action_mean = 0.9620328849028401
action_std = 0.19111675297670516

# normalization coefficients based on 8 games (include layer states)
state_space_means = np.array([1486.280883666978, 0.7177584227036928, 8.057975417426984, -0.0032841025876638214, -0.0035958567181175703, -0.0091047623648402, -0.02247905754647863, -0.013169226812600457, -0.01237572453419263, 0.0004242282056696699, 0.006223947237480189, 0.4657816433966583, 104.00552354799838, 0.9773403524834058, 0.6054016937514305])
state_space_stds = np.array([19881.36443950499, 0.45555367111140643, 7.446578343070469, 0.042706703714127395, 0.057142995431874025, 0.1053557243148667, 0.15276183436700566, 0.05295467287818651, 0.05693158815729483, 0.015152790450373455, 0.10062886537006414, 0.49882772985409507, 62.76831423403671, 0.8139823934799428, 1.092192534279478])
reward_mean = 0.7538862852782988
reward_std = 0.7399557592778772
action_mean = 0.9537651636530099
action_std = 0.2099932766898912

