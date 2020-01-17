import numpy as np

# simulation
g_timestep = 0.003

# tower
g_blocks_num = 54
scaler = 50
one_millimeter = 0.001 * scaler

# coordinate axes parameter
coordinate_axes_pos_x = -0.15 * scaler
coordinate_axes_pos_y = -0.15 * scaler
coordinate_axes_pos_z = 0
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