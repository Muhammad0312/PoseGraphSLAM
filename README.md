﻿# Hands-on Localization

##dependencies:

pip install shapely
pip install scipy

last update:

## HOW TO RUN:

**launch sim**
1. roslaunch pose-graph-slam turtlebot_hoi_circuit1.launch

**Launch manual control nodes**

2. roslaunch pose-graph-slam kobuki_keyboard_control.launch

**Launch SLAM**

4. rosrun pose-graph-slam integration.py
