# Hands-on Localization

##dependencies:

pip install shapely
pip install scipy

last update:


## How to Run

roslaunch pose-graph-slam test_kobuki_basic.launch
rosrun pose-graph-slam working.py
roslaunch pose-graph-slam kobuki_keyboard_control1.launch

## HOW TO RUN:

**launch sim**
1. roslaunch turtlebot_simulation turtlebot_hoi_circuit1.launch
or 
2. roslaunch turtlebot_simulation turtlebot_hoi_circuit2.launch

**Launch manual control nodes**

2. roslaunch pose-graph-slam kobuki_keyboard_control.launch

**Launch SLAM**

4. rosrun pose-graph-slam integration.py
