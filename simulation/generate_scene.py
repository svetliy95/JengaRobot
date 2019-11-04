block_x = 0.075
block_y = 0.025
block_z = 0.015
timestamp = 0.001
block_1st_layer_pos_x = 0.0
block_1st_layer_pos_y = 0.05
block_2nd_layer_pos_x = 0.05
block_2nd_layer_pos_y = 0.0

def generate_block(position, even):
    if even:
        pass
    pass


def generate_layer(even):
    pass
def generate_scene():


    MODEL_XML = f"""
    <?xml version="1.0" ?>
    <mujoco>
    
        <statistic extent="2" meansize=".05"/>
    
        <option timestep="0.0005" solver="CG" integrator="RK4" iterations="30" tolerance="1e-10" jacobian="sparse" cone="pyramidal"/>
    
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
            <geom name="ground" type="plane" size="0 0 1" pos="0 0 -1" quat="1 0 0 0" material="matplane" condim="1"/>
    
            <body name="block1" pos="0.0 0.051 0.015">
                <freejoint/>
                <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
                <camera euler="0 0 0" fovy="40" name="rgb" pos="0 0 0.025"></camera>
            </body>
            <body name="block2" pos="0 0 0.015">
                <freejoint/>
                <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
            </body>
            <body name="block3" pos="0.0 -0.051 0.015">
                <freejoint/>
                <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
            </body>
    
            <body name="block4" pos="0.051 0 0.045" euler="0 0 90">
                <freejoint/>
                <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
            </body>
            <body name="block5" pos="0 0 0.045" euler="0 0 90">
                <freejoint/>
                <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
            </body>
            <body name="block6" pos="-0.051 0 0.045" euler="0 0 90">
                <freejoint/>
                <geom mass="0.01467" pos="0 0 0" rgba="1 0 0 1" size="0.075 0.025 0.015" type="box"/>
            </body>
    
    
        </worldbody>
        <actuator>
        </actuator>
    </mujoco>
    """