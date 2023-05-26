import numpy as np
import matplotlib.pyplot as plt

'''
The sign of the angle has been changed from the pose vector
'''
def PointFeatureCompounding(point, pose):
    T = np.array([[np.cos(pose[2]), -np.sin(pose[2]), pose[0]],
                  [np.sin(pose[2]),  np.cos(pose[2]), pose[1]],
                  [       0,                0,          1   ]])
    
    P = np.array([[point[0]],
                  [point[1]],
                  [   1    ]])

    return (T @ P)[0:2, :] # Take first two components and ignore the third


def ToWorldFrame(state_vector, map):
    compounded_scans = []
    for i in range(0, len(map)):
        scan = []
        for point in map[i]:
            scan.append(PointFeatureCompounding(point, state_vector[i])[0:2].reshape(1, 2)[0])
        scan = np.array(scan)
        compounded_scans.append(scan)
    
    return compounded_scans


def scans_to_map(state_vector, map):
    # map = [S1, S2, S3, S4]
    # poses = [x1, x2, x3, x4]
    
    # convert from [x,y,z] to [[x,y,z]]
    state_vector = [[state_vector[i], state_vector[i+1], state_vector[i+2]] for i in range(0, len(state_vector) -3, 3)]
    
    compounded_scans = ToWorldFrame(state_vector, map)

    full_scan = []

    for scan in compounded_scans:
        for point in scan:
            full_scan.append([point[0],point[1],0])
    return full_scan

    
