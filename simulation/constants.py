# simulation
g_timestep = 0.002

# tower
g_blocks_num = 54
scaler = 20
one_millimeter = 0.001 * scaler

# coordinate axes parameter
coordinate_axes_pos_x = -0.15 * scaler
coordinate_axes_pos_y = -0.15 * scaler
coordinate_axes_pos_z = 0
coordinate_axes_width = 0.005 * scaler
coordinate_axes_height = 0.025 * scaler

# empirical data
block_length_mean = 0.0747722 * scaler
block_length_sigma = 0.000236 * scaler
block_width_mean = 0.02505 * scaler
block_width_sigma = 0.000225 * scaler
block_height_mean = 0.0146361 * scaler
block_height_sigma = 0.00003773 * scaler
block_mass_mean = 0.012865 * scaler**3
block_mass_sigma = 0.00212 * scaler**3

# vectors
x_unit_vector = [1, 0, 0]
y_unit_vector = [0, 1, 0]
z_unit_vector = [0, 0, 1]

# sensing
timesteps_transient = 75
force_threshold = 1000
displacement_threshold = 0.3

