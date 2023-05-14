from math import sin, cos
import numpy as np

def get_scan(scan_msg):
        
    # self.update_running = True
    ranges = np.array(scan_msg.ranges)
    angle_min = scan_msg.angle_min
    angle_increment = scan_msg.angle_increment
    
    num_points = len(ranges)
    angles = np.linspace(angle_min, angle_min + angle_increment * (num_points - 1), num_points)
    
    curr_scan = []
    for i in range(num_points):
        if ranges[i] < scan_msg.range_max and ranges[i] > scan_msg.range_min:
            x = ranges[i] * cos(angles[i])
            y = ranges[i] * sin(angles[i])
            curr_scan.append((x, y))
        
    curr_scan = np.array(curr_scan)

    return curr_scan