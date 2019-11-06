from numpy.random import normal

# global constants
scaler = 20
timestep = 0.01
# empirical data
block_length_mean = 0.0747722 * scaler
block_length_sigma = 0.000236 * scaler
block_width_mean = 0.02505 * scaler
block_width_sigma = 0.000225 * scaler
block_height_mean = 0.0146361 * scaler
block_height_sigma = 0.00003773 * scaler
block_mass_mean = 0.012865 * scaler**3
block_mass_sigma = 0.00212 * scaler**3



def generate_block(number, pos_sigma, angle_sigma, spacing):
    # TODO spacing automatic calculation

    if number % 6 < 3:  # even level
        x = 0
        y = -block_width_mean + (number % 3) * block_width_mean
        y += (number % 3) * spacing  # add spacing between blocks
        z = number//3 * block_height_mean
        angle_z = normal(0, angle_sigma)  # add disturbance to the angle
    else:  # odd level
        x = -block_width_mean + (number % 3) * block_width_mean
        x += (number % 3) * spacing  # add spacing between blocks
        y = 0
        z = number//3 * block_height_mean
        angle_z = normal(90, angle_sigma)  # rotate and add disturbance

    # add disturbance to mass, position and sizes
    mass = normal(block_mass_mean, block_mass_sigma)
    [x, y] = normal([x, y], [pos_sigma, pos_sigma])
    [block_size_x, block_size_y, block_size_z] = normal([block_length_mean/2, block_width_mean/2, block_height_mean/2],
                                                        [block_length_sigma, block_width_sigma, block_height_sigma])

    s = f'''
            <body name="block{number}" pos="{x} {y} {z+block_height_mean/2}" euler="0 0 {angle_z}">
                <freejoint/>
                <geom mass="{mass}" pos="0 0 0" class="block" size="{block_size_x} {block_size_y} {block_size_z}" type="box"/>
            </body>'''
    return s


def generate_scene(pos_sigma=0.0005,
                   angle_sigma=0.2,
                   spacing=block_width_sigma*3):

    blocks_xml = ""
    for i in range(54):
        blocks_xml += generate_block(i, pos_sigma, angle_sigma, spacing)

    MODEL_XML = f"""
    <?xml version="1.0" ?>
    <mujoco>
        
        <size nconmax="3200" njmax="8000"/>
    
        <statistic extent="2" meansize=".05"/>
        
        <default class="main">
            <default class="block">
                <geom rgba="0 1 0 1" condim="4"/>
            </default>
        </default>
    
        <option timestep="{timestep}" integrator="Euler" cone="elliptic" solver="Newton" o_solimp="0.999 0.999 0.01 0.5 2" o_solref="0 1"/>
    
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
    string = generate_scene()
    print(string)
    file = open("/home/bch_svt/.mujoco/mujoco200/model/jenga.xml", "w")
    file.write(string)