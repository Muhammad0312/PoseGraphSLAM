import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon
import time

'''
The sign of the angle has been changed from the pose vector
'''
def PointFeatureCompounding(point, pose):
    T = np.array([[np.cos(pose[2]), -np.sin(pose[2]), pose[0]],
                  [np.sin(pose[2]),  np.cos(pose[2]), pose[1]],
                  [       0,                0,           1   ]])
    
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


def OverlappingScans(state_vector, map,offset):
    H = []

    overlap_threshold = 80
    compounded_scans = map

    point_cloud_1 = compounded_scans[-1]
    start = 0
    # offset = 2
    if len(compounded_scans) > offset:
        start = len(compounded_scans) - offset
    
    for i in range( start, len(map) - 1):
        point_cloud_2 = compounded_scans[i]
        # Define a tolerance distance
        tolerance = 0.13

        # Build a KDTree for each point cloud

        #print the shape of the point cloud
        tree_1 = KDTree(point_cloud_1)
        tree_2 = KDTree(point_cloud_2)

        # Query each tree for the nearest neighbors of each point in the other tree
        dist_1, ind_1 = tree_1.query(point_cloud_2, distance_upper_bound=tolerance)
        dist_2, ind_2 = tree_2.query(point_cloud_1, distance_upper_bound=tolerance)

        # Find the indices of pairs of points that are close to each other in both clouds
        overlap_indices_1 = np.where(np.isfinite(dist_1))[0]
        overlap_indices_2 = np.where(np.isfinite(dist_2))[0]

        # Extract the overlapping points from each point cloud
        overlapping_points_1 = point_cloud_1[overlap_indices_2]
        overlapping_points_2 = point_cloud_2[overlap_indices_1]
        if (len(overlapping_points_1) / len(point_cloud_1))*100 >= overlap_threshold:
            H.append(i)

            # #______________________     IMAGE SAVING __________________________________
            # fig = plt.figure(figsize=(10, 5))
            # ax1 = fig.add_subplot(111)
            # ax1.scatter(point_cloud_1[:,0], point_cloud_1[:,1], c='blue', s=1)
            # ax1.scatter(point_cloud_2[:,0], point_cloud_2[:,1], c='red', s=1)
            # plt.savefig('/home/alamdar11/projects_ws/src/pose-graph-slam/src/saved_data/overlapping_images/image'+str(np.round(time.time(), 2))+'.png')
            # plt.close()
            # #_________________________________________________________________________

    return H

def OverlappingScansConvex(state_vector, map):

    state_vector = [[state_vector[i], state_vector[i+1], state_vector[i+2]] for i in range(0, len(state_vector), 3)]

    compounded_scans = ToWorldFrame(state_vector, map)
    points1 = compounded_scans[-1]
    H = []
    overlap_threshold = 100
    for i in range(0, len(compounded_scans) - 1):
        points2 = compounded_scans[i]

        # compute the convexhull
        hull1 = ConvexHull(points1)
        hull2 = ConvexHull(points2)

        # create Polygon objects for the two convex hulls
        poly1 = Polygon(points1[hull1.vertices])
        poly2 = Polygon(points2[hull2.vertices])

        # compute the intersection area
        intersection_area = poly1.intersection(poly2).area

        if (intersection_area/poly1.area)*100 >= overlap_threshold:
            H.append(i)

    return H
