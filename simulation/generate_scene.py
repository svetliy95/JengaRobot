from numpy.random import normal

# global constants
block_length_mean = 7.5
block_width_mean = 2.5
block_height_mean = 1.5
timestamp = 0.001
mass_mean = 12000


def generate_block(number, mass_sigma, size_sigma, pos_sigma, angle_sigma, spacing):
    # TODO spacing automatic calculation

    if number % 6 < 3:  # even level
        x = 0
        y = -block_width_mean + (number % 3) * block_width_mean
        y += (number % 3) * spacing
        z = number//3 * block_height_mean
        angle_z = normal(0, angle_sigma)
    else:  # odd level
        x = -block_width_mean + (number % 3) * block_width_mean
        x += (number % 3) * spacing
        y = 0
        z = number//3 * block_height_mean
        angle_z = normal(90, angle_sigma)

    # add disturbance
    mass = normal(mass_mean, mass_sigma)
    [x, y] = normal([x, y], [pos_sigma, pos_sigma])
    [block_size_x, block_size_y, block_size_z] = normal([block_length_mean/2, block_width_mean/2, block_height_mean/2], [size_sigma for x in range(3)])

    s = f'''
            <body name="block{number}" pos="{x} {y} {z}" euler="0 0 {angle_z}">
                <freejoint/>
                <geom mass="{mass}" pos="0 0 0" rgba="1 0 0 1" size="{block_size_x} {block_size_y} {block_size_z}" type="box"/>
            </body>
'''
    return s
def generate_scene(mass_sigma = 0.002, size_sigma = 0.0005, pos_sigma = 0.001, angle_sigma = 2, spacing = 0.001):

    blocks_xml = ""
    for i in range(54):
        blocks_xml += generate_block(i, mass_sigma, size_sigma, pos_sigma, angle_sigma, spacing)

    MODEL_XML = f"""
    <?xml version="1.0" ?>
    <mujoco>
        
        <size nconmax="3200" njmax="8000"/>
    
        <statistic extent="2" meansize=".05"/>
    
        <option timestep="{timestamp}" solver="Newton" integrator="Euler" iterations="30" tolerance="1e-10" jacobian="sparse" cone="pyramidal"/>
    
        <visual>
            <rgba haze="0.15 0.25 0.35 1"/>
            <quality shadowsize="2048"/>
            <map stiffness="700" shadowscale="0.5" fogstart="10" fogend="15" zfar="40" haze="0.3"/>
        </visual>
    
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="512"/> 
            <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2" 
                width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
            <texture name="texsponge" type="2d" file="/home/bch_svt/cartpole/simulation/sponge.png"/>   
            <texture name="texcarpet" type="2d" file="/home/bch_svt/cartpole/simulation/carpet.png"/>  
    
            <texture name="texmarble" type="cube" file="/home/bch_svt/cartpole/simulation/marble.png"/>s
    
            <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="1 1" texuniform="true"/>
            <material name="matcarpet" texture="texcarpet"/>
            <material name="matsponge" texture="texsponge" specular="0.3"/>
            <material name="matmarble" texture="texmarble" rgba=".7 .7 .7 1"/>
        </asset>
    
        <worldbody>
            <light directional="true" diffuse=".4 .4 .4" specular="0.1 0.1 0.1" pos="0 0 5.0" dir="0 0 -1" castshadow="false"/>
            <light directional="true" diffuse=".6 .6 .6" specular="0.2 0.2 0.2" pos="0 0 4" dir="0 0 -1"/>
            <geom name="ground" type="plane" size="0 0 1" pos="0 0 0" quat="1 0 0 0" material="matplane" condim="1"/>
            
            {blocks_xml} 
    
        </worldbody>
        <actuator>
        </actuator>
    </mujoco>
    """
    return MODEL_XML

if __name__ == "__main__":
    print(generate_scene(0, 0, 0, 0, 0))