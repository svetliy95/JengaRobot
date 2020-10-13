import numpy as np
from pyquaternion import Quaternion

# simulation
g_timestep = 0.005

# tower
g_blocks_num = 48
g_blocks_max = 54
flipping_threshold = 30  # in degrees
# scaler = 50
# one_millimeter = 0.001 * scaler
scaler = 1000
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
timeout_step = 90

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
state_space_means_real_robot = np.array([0.19520652173913025, 0.7177584227036928, 8.057975417426984, -0.0032841025876638214, -0.0035958567181175703, -0.0091047623648402, -0.02247905754647863, -0.013169226812600457, -0.01237572453419263, 0.0004242282056696699, 0.006223947237480189, 0.4657816433966583, 104.00552354799838, 0.9773403524834058, 0.6054016937514305])
state_space_stds_real_robot = np.array([0.1838663584877718, 0.45555367111140643, 7.446578343070469, 0.042706703714127395, 0.057142995431874025, 0.1053557243148667, 0.15276183436700566, 0.05295467287818651, 0.05693158815729483, 0.015152790450373455, 0.10062886537006414, 0.49882772985409507, 62.76831423403671, 0.8139823934799428, 1.092192534279478])
reward_mean = 0.7538862852782988
reward_std = 0.7399557592778772
action_mean = 0.9537651636530099
action_std = 0.2099932766898912

# real robot, 9 features
state_space_means_real_robot = np.array([0.10157205720572064, 1.619317931793178, 9.016795012834615, 0.0802000645576744, 0.18506699112818664, 0.06738466679637432, 167.89254980375293, 0.9823982398239824, 0.8217821782178217])
state_space_stds_real_robot = np.array([0.14969758488355905, 1.583845527100756, 8.034401777720147, 0.12682736963599048, 0.2626609181054224, 0.4439298959783069, 63.437185363129494, 0.8389043956729575, 1.2893439513725289])
reward_mean_real_robot = 1.1911890666102654
reward_std_real_robot = 2.3456955369081576
action_mean_real_robot = 0.8866886688668867
action_std_real_robot = 0.31697298523684203

# real robot, 7 features
state_space_means_real_robot = np.array([0.10157205720572064, 1.619317931793178, 9.016795012834615, 0.0802000645576744, 0.18506699112818664, 0.06738466679637432, 167.89254980375293])
state_space_stds_real_robot = np.array([0.14969758488355905, 1.583845527100756, 8.034401777720147, 0.12682736963599048, 0.2626609181054224, 0.4439298959783069, 63.437185363129494])
reward_mean_real_robot = 1.1911890666102654
reward_std_real_robot = 2.3456955369081576
action_mean_real_robot = 0.8866886688668867
action_std_real_robot = 0.31697298523684203

# cameras
cam1_serial = '22917552'
cam1_mtx = np.array([[4.47099400e+03, 0.00000000e+00, 2.10612420e+03],
       [0.00000000e+00, 4.46920081e+03, 1.52419164e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
cam1_dist = np.array([[-0.01028824, -0.03225693, -0.00041508,  0.00311179,  0.21768256]])

cam1_mtx_11cm = np.array([[4.19754548e+03, 0.00000000e+00, 2.05859765e+03],
       [0.00000000e+00, 4.20065530e+03, 1.54661619e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
cam1_dist_11cm = np.array([[-0.03946547, -0.10121212,  0.00088056,  0.00040551,  0.32114937]])
image_scale = 0.5

cam1_mtx_11cm_scaled_05 = np.array([[2.09891285e+03, 0.00000000e+00, 1.02867787e+03],
       [0.00000000e+00, 2.10027551e+03, 7.72510205e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
cam1_dist_11cm_scaled_05 = np.array([[-0.04040886, -0.08988447,  0.00091706,  0.00029727,  0.2824488 ]])

cam1_mtx_11cm_2 = np.array([[4.22987816e+03, 0.00000000e+00, 2.06704818e+03],
       [0.00000000e+00, 4.22699160e+03, 1.55592011e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

cam1_dist_11cm_2 = np.array([[-0.04208958, -0.04683562,  0.00067192,  0.00087862,  0.22510104]])

cam1_mtx_11cm_3 = np.array([[4.21878895e+03, 0.00000000e+00, 2.04993997e+03],
       [0.00000000e+00, 4.22408301e+03, 1.54796393e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
cam1_dist_11cm_3 = np.array([[-0.04350285, -0.03094388,  0.00024685, -0.00073692,  0.1379812 ]])

cam2_serial = '22919001'
cam2_mtx = np.array([[2.10038867e+03, 0.00000000e+00, 1.59039780e+03],
       [0.00000000e+00, 2.10005648e+03, 1.05518348e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
cam2_dist = np.array([[-0.00600466,  0.02840844,  0.00043355,  0.00111766, -0.04208812]])

phone_cam_mtx = np.array([[934.42689912, 0., 637.96645563],
       [0., 934.56635613, 491.63680933],
       [0., 0., 1.]])
phone_cam_dist = np.array([[0.16284682, -0.69159724,  0.00360451,  0.00146134,  0.82432896]])

# full size image
phone_cam_mtx = np.array([[2.91991255e+03, 0.00000000e+00, 2.01644404e+03],
       [0.00000000e+00, 2.91841106e+03, 1.53419836e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
phone_cam_dist = np.array([[ 0.11169081, -0.42087882,  0.00243412,  0.00184474,  0.41748723]])

# tags detection
detection_method = 'apriltag'  # 'apriltag' or 'aruco'
detection_threads = 8
quad_decimate1 = 3.0  # 3
quad_sigma1 = 1.3
decode_sharpening1 = 1.0
quad_decimate2 = 1.5  # 1.5
quad_sigma2 = 1.3
decode_sharpening2 = 1.0


# robots
buffer_size = 1024
right_robot_ip = '192.168.10.103'
right_robot_port = 10002
right_gripper_ip = '192.168.10.2'
left_robot_ip = '192.168.10.101'
left_robot_port = 10002
left_gripper_ip = '192.168.10.1'
real_tag_pos = np.array([0, 0, 0])
right_robot_home_position_world = np.array([158.613, -156.59, 95.871, 0.0, 0.0, 135])
right_robot_home_position_world = np.array([78.632, -19.887, 320, 0, 0, -45])
zwischenablage_place_pose = np.array([298.443, 88.613, 76.435, 34.864, -29.77 , -12.606])
zwischenablage_place_pos = np.array([298.443, 88.613, 76.435])
zwischenablage_place_quat = Quaternion(0.9249325768880927, 0.2608642425642888, -0.2753860975559344, -0.024740097724662957)
zwischenablage_take_pose = np.array([257.94 ,  90.479,  54.222, -35.076, -30.15 , 100.069])
zwischenablage_take_pos = zwischenablage_take_pose[:3]
zwischenablage_take_quat = Quaternion(0.6514593292525659, 0.0031738377130947626, -0.3823000037465868, 0.6553147153996525)  # euler2quat(zwischenablage_take_pose[3:], degrees=True)
right_robot_spring_constant = 0.075  # N/mm

# play
loose_block_height_threshold = 0.05  # in mm




# right_robot_home_position_world = np.array([218.737, -216.677, 95.866, 0.0, 0.0, 135])
right_robot_taster_offset = -67  # in mm
stopover_height = 320

read_force_wait = 1.0  # seconds

# simulation localization errors
x_error_dist_params = (1.2655901036253412, -0.15175142356915827, -0.1153019687491872, 0.3251173548429831)
y_error_dist_params = (0.8473677321389348, -0.012511120229234098, -0.08008866438405154, 0.24949697819455977)
z_error_dist_params = (0.2685601868012577, -0.13014166151038362, 0.01544363740802453, 0.3718678598541926)
a_error_dist_params = (0.008908198635120112, 0.00796167065070429, 1.667838518095881e-05, 0.0002170482803487573)
b_error_dist_params = (1.2818516675852127, -0.05294795207878875, 5.512792068010095e-05, 0.0007857210858762351)
c_error_dist_params = (2.8803303722483795, 0.14453792763159595, -0.00013869870560313506, 0.0015285142373694157)
d_error_dist_params = (1.6607936208328635, 0.38880334458681665, -0.0007712423289701057, 0.005872380446434848)

# block selection ML
block_selection_mean =  np.array([14.69215052, 14.84400322, 14.87192898,  0.33333333,  0.33333333,  0.33333333])
block_selection_mean =  np.array([14.68956037, 14.85490134, 14.87550177,  0.33333333,  0.33333333,  0.33333333])
block_selection_mean =  np.array([14.67820434, 14.8338043,  14.87485554,  0.33333333,  0.33333333,  0.33333333])

block_selection_std = np.array([0.3105194,  0.32565065, 0.31335974, 0.47140452, 0.47140452, 0.47140452])
block_selection_std = np.array([0.3271648,  0.32295477, 0.29890007, 0.47140452, 0.47140452, 0.47140452])
block_selection_std = np.array([0.30570986, 0.32289002, 0.31656156, 0.47140452, 0.47140452, 0.47140452])

