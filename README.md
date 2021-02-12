# Instructions
### Robot starting 
1. Switch the main robot controller on
2. Restart the controlling PLC

### Robot playing start
1. Switch on servos
2. Start the execution of the program *jenga* on the robot controller.
3. Execute the script play.py (you must confirm a tower reset by pressing any key in the console)

:exclamation: Check camera parameters in pylonViewer before start. Unsuitable parameters may lead to huge localization errors. :exclamation:

### Custom controlling
Create an instance of the class _jenga_env_ from _read_jenga_env.py_ and use its methods for robot controlling.

### Blocks calibration
:bangbang: Blocks are already calibrated. Do this only if you want to recalibrate or use another set of jenga blocks.

For the block calibration place the calibration device near the camera. Specify necessary paths and the number of tags in the _blocks_calibration.py_ and start it. The script will make 3 images for each tag and then average the poses of the block relative to the calibration device. So change the pose of the calibration device together with block ralative to the camera for each capture. To trigger a capture press any key on the keyboard. While blocks are calibrated the script writes calibration values into a specified file.

### Camera parameters
Use pylonViewer for this. You need to set exposure timesevery time you restart the computer.

Rough parameters for the cameras:
|Camera|Exposure Auto|Auto Target Brightness|
|-|-|-|
|acA4024-29um|Continuous|0.45|
|acA3088-57um|Continuous|0.45|

### Gripper referencing
You can access the grippers using a browser at adresses 192.168.10.1 and 192.168.10.2
There you can reference the gripper (set a zero point for its fingers).

### Coordinate system calibration
For Robot initializing you need to define a world coordinate system for it. It can be done using 3 points measured with the robot itself. For this:
1. Attach the special tool to the gripper
2. Make shure the gripper is perpendicular to the floor
3. Make shure the right tool is selected on the robot controller (Tool 1)
4. Touch with the tip of the tool and measure positions of 3 points (for getting point positions you can use method Robot.get_pose()):
    * World coordinate system origin
    * Some point on the x-axis
    * Some point on the y-axis
5. Initialize with the measured coordinates the following variables in the constants.py:
    * right_robot_x_axis_point
    * right_robot_y_axis_point
    * right_robot_origin_point
    




# Most important files
| File/Directory | Description |
|-|-|
| ./play.py | Execute this script to start robot playing using ML. |
| ./real_robot_env.py | This script contains jenga_env class that is representing jenga environment with the real robot. |
| ./constants.py | This file contains all parameter for the whole system (incl. simulation and real robot environment) |
| ./tower.py | Implements a class representing a jenga tower. |
| ./cv/blocks_calibration.py | This script is used for calibration of jenga blocks. |
| ./cv/localization_visualization.py | Very important script! It visualizes real jenga tower in virtual environment. Simply start it and check that the block localization works correctly.  Green blocks mean that they are visible from both cameras. Yellow - the blocks are visible only from one camera. Red - the block is gone from the visual fields of both cameras. |
| ./data/block_sizes.json | Dimensions of all jenga blocks. |
| ./data/corrections.json | File with corrections for each apriltag computed during calibration (using ./cv/blocks_calibration.py). |
| ./training_data/loose_blocks_data.csv | Training data for the network for loose block finding. The first 3 numbers are the block ids. The last number is the block configuration (see corresponding master thesis). |
| ./data/measured_poses.json | Temporal file used for calibration. |
| ./models/ | Contains pre-trained models. |
| ./rl/collect_expert_data.py | Script for collecting training data. Simply replace real robot environment with the simulation environment for data collection in the simulation. |
| ./rl/pretrain_real_robot.py | Script for training (behavioral cloning) of the "pushing" model for the real robot. |
| ./rl/pretrain_simulation.py | Script for training (behavioral cloning) of the "pushing" model for the simulated environment. It can run multiple simulations simultaneously. |
| ./rl/process_expert_data.py | Tool needed for training data normalization. |
| ./rl/train_loose_blocks_finding.py | Script for training of the NN for finding loose blocks. |
| ./robots/robot.py | Contains the class *Robot* allowing easy communication with the robot. |
| ./robots/gripper.py | Contains the class Gripper implementing the communication protocol with the Schunk gripper. |
| ./simulation/jenga.py | Contains class implementing simulated jenga environment. |
| ./training_data/ | Training data used for training models located in ./models/ |
| ./utils/transformations.py | Functions for coordinate transformations. |
| ./utils/utils.py | Mostly functions for working with vectors and quaternions. |
| ./robots/MELFA/Jenga.zip | RT ToolBox2 WorkSpace. |
| ./robots/jenga_robot.txt | The program running on the robot. |
| ./resources/3d_models/ | 3d models used for printing. |
| ./resources/A3_ref_tag.png | Image with the reference tag. |


# Things to pay attention to
* Reference the gripper after each restart if it. Otherweise the gripper will not work.
* Set proper camera parameters in pylonViewer after each PC restart. They will be not saved. (Further passing of the camera parameter from the python side can be implemented. This feature is not implemented yet.)
* :bangbang: Do not mix two sets of jenga blocks!!! They share the same ids. Mixing blocks from two different sets will corrupt blocks calibration.
* After replacement of any tag (reference tag or tags on the blocks) adapt tag sizes all over the code. Otherwise localization won't work correctly.
* Adapt the variable _scaler_ in _constants.py_ while switching from real system to simulation and vice versa. For the simulation the scaler must be 50 and for the real environment 1000.
* After last code refactoring (12.02.2021) it can happen that the paths (or module imports) used in some scripts are not valid anymore. This can lead to errors. In this case try to change these paths/imports.
 
# Recommendations
* Use PyCharm. You can open the repository folder as a PyCharm project.
* Start *localization_visulization.py* before robot playing start. Check that the blocks are correctly recognized and localized. In the best case all blocks must be colored green and there must be not too much positions jitter.
* For the first run lower the speed on the robot controller.
* You may want to play with *apriltags detector* parameters. They schould be optimal for current setup (11.02.2021), but should be possibly adapted after lense, camera or distance from camera to the turm change.

  

# Issues
### Robot not responding
```python
Traceback (most recent call last):
  File "/home/bch_svt/cartpole/simulation/robots/robot.py", line 409, in <module>
    pose_robot = r1.get_world_pose(degrees=True)
  File "/home/bch_svt/cartpole/simulation/robots/robot.py", line 148, in get_world_pose
    pose = self.get_pose()
  File "/home/bch_svt/cartpole/simulation/robots/robot.py", line 144, in get_pose
    pos = np.array(list(map(float, pos)))
ValueError: could not convert string to float:
```
Solution: restart both main controller and the controlling PLC.
 
</blockquote>

### Unable to connect to camera
```python
Traceback (most recent call last):
  File "/home/bch_svt/cartpole/cv/localization_visualization.py", line 442, in <module>
    cam1 = Camera(cam1_serial, cam1_mtx, cam1_dist)
  File "/home/bch_svt/cartpole/cv/camera.py", line 46, in __init__
    device_instace = tlFactory.CreateDevice(device)
  File "/home/bch_svt/anaconda3/envs/rl/lib/python3.7/site-packages/pypylon/pylon.py", line 1456, in CreateDevice
    return _pylon.TlFactory_CreateDevice(self, *args)
_genicam.RuntimeException: Failed to open device '2676:ba02:4:5:3' for XML file download. Error: 'Device is exclusively opened by another client.' : RuntimeException thrown (file 'PylonUsbTL.cpp', line 474)
```
Check that non of the cameras are connected with the pylonViewer. Kill other python processes that may use cameras.

### In case of "GLEW initalization error: Missing GL version" 
Check https://github.com/openai/mujoco-py/issues/408#issuecomment-566796804

### Missing path to your environment variable.
```python
Exception: 
Missing path to your environment variable. 
Current values LD_LIBRARY_PATH=
Please add following line to .bashrc:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/bch_svt/.mujoco/mujoco200/bin
```

Set following environment variables or pass them as parameters while executing the script:
```
PYTHONUNBUFFERED=1;LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/x86_64-linux-gnu/libGL.so;LD_LIBRARY_PATH=/home/bch_svt/.mujoco/mujoco200/bin:/usr/lib/nvidia-000:/usr/lib/x86_64-linux-gnu/libGL.so
```
If using PyCharm paste this string into the line "environment parameters" by "Run/Debug Configurations" of the script.


# Configuration
If someone was able to configure this project on an another computer please fill this section with the configuration instructions.

```sh
$ sudo apt-get install libosmesa6-dev
$ sudo apt install patchelf
```


