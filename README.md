# Hands-on Localization

##dependencies:

pip install shapely
pip install scipy

last update:

## HOW TO RUN:

**launch sim**
1. roslaunch turtlebot_simulation turtlebot_hoi_circuit1.launch

**Launch manual control nodes**

2. rosrun pose-graph-slam kobuki_teleop.py (takes key command and publishes v,w)

3. rosrun pose-graph-slam inverse_kinematics_diff_drive.py (takes v,w and publishes wheel velocities)

**Launch SLAM**

4. rosrun pose-graph-slam pose_graph_SLAM.py
