import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon
'''
pip install shapely
pip install scipy

'''
'''
The sign of the angle has been changed from the pose vector
'''
def PointFeatureCompounding(point, pose):
    T = np.array([[np.cos(-pose[2]), -np.sin(-pose[2]), pose[0]],
                  [np.sin(-pose[2]),  np.cos(-pose[2]), pose[1]],
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


def OverlappingScans(state_vector, map):
    compounded_scans = ToWorldFrame(state_vector, map)
    H = []
    overlap_threshold = 50

    point_cloud_1 = compounded_scans[-1]
    for i in range(0, len(compounded_scans) - 1):

        point_cloud_2 = compounded_scans[i]
        # Define a tolerance distance
        tolerance = 0.3

        # Build a KDTree for each point cloud
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


        # # Print the overlapping percentage
        # print("Overlapping points in cloud 1:", (len(overlapping_points_1)/len(point_cloud_1))*100, '%')
        # print("Overlapping points in cloud 2:", (len(overlapping_points_2)/len(point_cloud_2))*100, '%')

        '''Uncomment to visulaize things'''
        # fig = plt.figure(figsize=(10, 5))
        # ax1 = fig.add_subplot(projection='3d')
        # ax1.scatter(point_cloud_1[:,0], point_cloud_1[:,1], c='blue')
        # ax1.scatter(point_cloud_2[:,0], point_cloud_2[:,1], c='red')
        # ax1.set_xlabel('X')
        # ax1.set_ylabel('Y')
        # ax1.set_zlabel('Z')
        # ax1.set_title('Original Point Cloud')
        # plt.show()
    return H

def OverlappingScansConvex(state_vector, map):
    compounded_scans = ToWorldFrame(state_vector, map)

    points1 = compounded_scans[-1]
    H = []
    overlap_threshold = 50

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
        # print(intersection_area/poly1.area)

        if (intersection_area/poly1.area)*100 >= overlap_threshold:
            H.append(i)


        '''Uncomment to visulaize things'''
        # plt.plot(points1[hull1.vertices,0], points1[hull1.vertices,1], 'r--', lw=2)
        # # plt.plot(points1[hull1.vertices[0],0], points1[hull1.vertices[0],1], 'ro')

        # plt.plot(points2[hull2.vertices,0], points2[hull2.vertices,1], 'b--', lw=2)
        # # plt.plot(points2[hull2.vertices[0],0], points2[hull2.vertices[0],1], 'ro')
        # plt.show()

    return H




scan1 = []
scan2 = []
scan3 = []

with open('scan1.txt', 'r') as f:
    content = f.readlines()
    for line in content:
        coordinates = line.split()
        scan1.append([float(coordinates[0]), float(coordinates[1])])


with open('scan2.txt', 'r') as f:
    content = f.readlines()
    for line in content:
        coordinates = line.split()
        scan2.append([float(coordinates[0]), float(coordinates[1])])


with open('scan3.txt', 'r') as f:
    content = f.readlines()
    for line in content:
        coordinates = line.split()
        scan3.append([float(coordinates[0]), float(coordinates[1])])

scan1 = np.array(scan1)
scan2 = np.array(scan2)
scan3 = np.array(scan3)

Map = [scan1, scan2, scan3]

P1 = [8.952061389220677316e-07, -6.361812210453066930e-11, -1.044700073057142620e-05]
P2 = [5.882918886136125416e-03, -7.026800482131475081e-03, -1.641024133333284230e+00]
P3 = [2.883825026077073139e-01, -1.402247462826852198e-01, -9.072827003143387747e-01]
state_vector = [P1, P2, P3]

print(OverlappingScans(state_vector, Map))
# print(OverlappingScansConvex(state_vector, Map))

