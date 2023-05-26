import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time

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



def ScanToWorldFrame(pose, scan1):
    # compounded_scans = []
    scan = []
    for point in scan1:
        scan.append(PointFeatureCompounding(point, pose)[0:2].reshape(1, 2)[0])
    scan = np.array(scan)
    # compounded_scans.append(scan)
    return scan

def ScanToWorldFrame2(xk, scan1):

    augmented_array = np.ones((scan1.shape[0], 4))
    augmented_array[:,:-1] = scan1
    # compute the transformation matrix btw between the robot and the lidar
    wTr = np.array([[np.cos(xk[2]), -np.sin(xk[2]), 0, xk[0] ], 
            [np.sin(xk[2]), np.cos(xk[2]), 0, xk[1]] , 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]])

    # Left-multiply the augmented array by the homogeneous matrix
    transformed_array = np.dot(wTr, augmented_array.T).T

    # Select all rows and all but the last column
    transformed_array = transformed_array[:, :-1]

    #print the shape of transformed array

    return transformed_array
    

# def ScanMatching(source_points, target_points):
#     """
#     Perform scan matching using Open3D.
    
#     Arguments:
#     source_points -- Nx3 array of source points (3D LiDAR scan)
#     target_points -- Nx3 array of target points (reference 3D LiDAR scan)
    
#     Returns:
#     transformation -- 4x4 transformation matrix
#     """

#     # Create Open3D point cloud objects
#     source_cloud = o3d.geometry.PointCloud()
#     source_cloud.points = o3d.utility.Vector3dVector(source_points)
#     target_cloud = o3d.geometry.PointCloud()
#     target_cloud.points = o3d.utility.Vector3dVector(target_points)

#     color = np.array([1.0, 0.0, 0.0])  # Set the desired color (here, red)
#     source_cloud.paint_uniform_color(color)

#     '''Visualization purposes'''

#     # p1 = np.copy(np.asarray(target_cloud.points))
#     # p2 = np.copy(np.asarray(source_cloud.points))
    

#     # x1 = p1[:, 0]
#     # y1 = p1[:, 1]

#     # x2 = p2[:, 0]
#     # y2 = p2[:, 1]

#     '''Careful not for visualization purposes'''
#     # Perform registration
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         source_cloud, target_cloud, max_correspondence_distance=0.1)
    
#     '''=============================================================='''
    
#     '''Debugging purposes'''

#     aligned_pcd1 = source_cloud.transform(reg_p2p.transformation)
#     p3 = np.asarray(aligned_pcd1.points)

#     # x3 = p3[:, 0]
#     # y3 = p3[:, 1]
    
#     # fig = plt.figure(figsize=(15, 5))
#     # ax1 = fig.add_subplot(121)
#     # ax1.scatter(x1, y1, c='blue', s=1)
#     # ax1.scatter(x2, y2, c='red', s=1)
#     # ax1.set_title("Orignal Scans")
#     # ax2 = fig.add_subplot(122)
#     # ax2.scatter(x1, y1, c='blue', s=1)
#     # ax2.scatter(x3, y3, c='red', s=1)
#     # ax2.set_title("Aligned Scans")
#     # plt.savefig('/home/mawais/catkin_ws/src/pose-graph-slam/src/saved_data/o3d_icp/image'+str(np.round(time.time(), 2))+'.png')
#     # plt.close()
#     '''======================================================='''
    

#     # Visualize the aligned point clouds
#     # o3d.visualization.draw_geometries([aligned_pcd1, target_cloud])

#     return reg_p2p.transformation, p3

def ICP(MatchedScan, CurrentScan, initial_guess): #, MatchedVp, CurrentVp

    '''Debugging Purposes'''

    # x1 = np.copy(MatchedScan[:,0])
    # y1 = np.copy(MatchedScan[:,1])
    # x2 = np.copy(CurrentScan[:,0])
    # y2 = np.copy(CurrentScan[:,1])
    '''==================================='''

    # state_vector = [MatchedVp, CurrentVp]
    # Map = [MatchedScan, CurrentScan]

    # compounded_scans = ToWorldFrame(state_vector, Map)

    temp_column = np.zeros(MatchedScan.shape[0])
    source_points = np.hstack((MatchedScan, temp_column.reshape(-1, 1)))

    temp_column = np.zeros(CurrentScan.shape[0])
    target_points = np.hstack((CurrentScan, temp_column.reshape(-1, 1)))

    # Create Open3D point cloud objects
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)
    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)

    color = np.array([1.0, 0.0, 0.0])  # Set the desired color (here, red)
    source_cloud.paint_uniform_color(color)



    #TODO: calculate the initial guess for the transformation matrix

    # #create a 4x4 transformation matrix out of the initial guess
    initial_guess = np.array([[np.cos(initial_guess[2]), -np.sin(initial_guess[2]), 0, initial_guess[0]],   
                                [np.sin(initial_guess[2]), np.cos(initial_guess[2]), 0, initial_guess[1]],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])


    #convert initial guess to float64
    initial_guess = initial_guess.astype(np.float64)

    # # # Perform registration
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     target_cloud, source_cloud, max_correspondence_distance=0.1)
    
    # Compute normal vectors for target point cloud
    # source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    #     # Perform registration
    reg_p2p = o3d.pipelines.registration.registration_icp(
        target_cloud, source_cloud, 3, initial_guess,
         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))
    
    
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    # target_cloud, source_cloud, 3, initial_guess,
    # o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    # o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))



    transformation = reg_p2p.transformation

    '''Debugging purposes'''
    # aligned_pcd1 = source_cloud.transform(transformation)
    # p3 = np.asarray(aligned_pcd1.points)

    # # Visualize the aligned point clouds
    # # o3d.visualization.draw_geometries([aligned_pcd1, target_cloud])

    # x3 = p3[:, 0]
    # y3 = p3[:, 1]


    # fig = plt.figure(figsize=(15, 5))
    # ax1 = fig.add_subplot(111)
    # ax1.scatter(x1, y1, c='blue', s=1)
    # ax1.scatter(x2, y2, c='red', s=1)
    # ax1.set_title("Orignal Scans")
    # ax2 = fig.add_subplot(122)
    # ax2.scatter(x2, y2, c='blue', s=1)
    # ax2.scatter(x3, y3, c='red', s=1)
    # ax2.set_title("Aligned Scans")
    # plt.savefig('/home/joseph/turtlebot_ws/src/pose-graph-slam/src/saved_data/o3d_icp/image'+str(np.round(time.time(), 2))+'.png')
    # plt.close()

    # Extract translation and rotation
    translation = transformation[0:2, 3]
    theta = np.arctan2(transformation[1, 0], transformation[0, 0])
    return [translation[0], translation[1], theta]

def PoseCompounding(NewViewpointInvertedPose, PreviousViewpointPose):

    x1 = NewViewpointInvertedPose[0]
    y1 = NewViewpointInvertedPose[1]
    theta1 = NewViewpointInvertedPose[2]

    x2 = PreviousViewpointPose[0]
    y2 = PreviousViewpointPose[1]
    theta2 = PreviousViewpointPose[2]

    x = x1 + x2*np.cos(theta1) - y2*np.sin(theta1)
    y = y1 + x2*np.sin(theta1) + y2*np.cos(theta1)
    theta = theta1 + theta2
    CompoundedPose = np.array([x, y, theta])

    return CompoundedPose

def PoseInversion(Pose):
    '''
    Pose = [x, y, theta]
    '''
    Coordinates = np.array([Pose[0], Pose[1]])
    InverseRotation = np.array([[np.cos(Pose[-1]),  np.sin(Pose[-1])],
                                [-np.sin(Pose[-1]), np.cos(Pose[-1])]])
    InvertedCoordinates = -(InverseRotation @ Coordinates)

    InvertedPose = np.array([InvertedCoordinates[0], InvertedCoordinates[1], -Pose[-1]])
    return InvertedPose


# scan1 = []
# scan2 = []
# scan3 = []

# with open('pose-graph-slam/src/overlapping thing/scan1.txt', 'r') as f:
#     content = f.readlines()
#     for line in content:
#         coordinates = line.split()
#         scan1.append([float(coordinates[0]), float(coordinates[1])])


# with open('pose-graph-slam/src/overlapping thing/scan2.txt', 'r') as f:
#     content = f.readlines()
#     for line in content:
#         coordinates = line.split()
#         scan2.append([float(coordinates[0]), float(coordinates[1])])


# with open('pose-graph-slam/src/overlapping thing/scan3.txt', 'r') as f:
#     content = f.readlines()
#     for line in content:
#         coordinates = line.split()
#         scan3.append([float(coordinates[0]), float(coordinates[1])])

# scan1 = np.array(scan1)
# scan2 = np.array(scan2)
# scan3 = np.array(scan3)

# Map = [scan1, scan2, scan3]

# P1 = [8.952061389220677316e-07, -6.361812210453066930e-11, -1.044700073057142620e-05]
# P2 = [5.882918886136125416e-03, -7.026800482131475081e-03, -1.641024133333284230e+00]
# P3 = [2.883825026077073139e-01, -1.402247462826852198e-01, -9.072827003143387747e-01]

# state_vector = [P1, P2, P3]

# # start_time = time.time()
# ICP(scan1, scan2, P1, P2)
# end_time = time.time()
# print('Execution time: ',end_time-start_time)


# compounded_scans = ToWorldFrame(state_vector, Map)

# temp_column = np.zeros(compounded_scans[0].shape[0])
# scan1 = np.hstack((compounded_scans[0], temp_column.reshape(-1, 1)))

# temp_column = np.zeros(compounded_scans[2].shape[0])
# scan2 = np.hstack((compounded_scans[2], temp_column.reshape(-1, 1)))

# transformation = scan_matching(scan1, scan2)
# print("Transformation matrix:")
# print(transformation)

'''Uncomment to visulaize things'''
# fig = plt.figure(figsize=(10, 5))
# ax1 = fig.add_subplot(projection='3d')
# ax1.scatter(scan1[:,0], scan1[:,1], scan1[:,2], c='blue')
# ax1.scatter(scan2[:,0], scan2[:,1], scan2[:,2], c='red')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# ax1.set_title('Original Point Cloud')
# plt.show()