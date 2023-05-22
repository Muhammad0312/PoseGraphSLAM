import open3d as o3d
import numpy as np

scan1 = []
scan2 = []
scan3 = []

with open('/home/mawais/catkin_ws/src/pose-graph-slam/src/saved_data/scan0.txt', 'r') as f:
    content = f.readlines()
    for line in content:
        coordinates = line.split()
        scan1.append([float(coordinates[0]), float(coordinates[1])])


with open('/home/mawais/catkin_ws/src/pose-graph-slam/src/saved_data/scan1.txt', 'r') as f:
    content = f.readlines()
    for line in content:
        coordinates = line.split()
        scan2.append([float(coordinates[0]), float(coordinates[1])])


# with open('/home/mawais/catkin_ws/src/pose-graph-slam/src/saved_data/scan2.txt', 'r') as f:
#     content = f.readlines()
#     for line in content:
#         coordinates = line.split()
#         scan3.append([float(coordinates[0]), float(coordinates[1])])

scan1 = np.array(scan1)
scan2 = np.array(scan2)
scan3 = np.array(scan3)

Map = [scan1, scan2, scan3]


# P1 = [0.0, 0.0, 0.0]
# P2 = [1.344714218718267781e-02, -4.710644820079053110e-04, -5.862067833890714178e-04]
# P3 = [2.081233576092281212e-02, -4.776966664179508812e-04, -9.052017214770224484e-04]
# state_vector = [P1, P2, P3]




target_cloud = o3d.geometry.PointCloud()
target_cloud.points = o3d.utility.Vector3dVector(np.hstack((scan1, np.zeros((scan1.shape[0], 1)))))

source_cloud = o3d.geometry.PointCloud()
source_cloud.points = o3d.utility.Vector3dVector(np.hstack((scan2, np.zeros((scan2.shape[0], 1)))))

color = np.array([1.0, 0.0, 0.0])  # Set the desired color (here, red)
source_cloud.paint_uniform_color(color)


o3d.visualization.draw_geometries([source_cloud, target_cloud])

# Estimate normals
source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Compute FPFH features
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(target_cloud, o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100))

# Set registration method and parameters
registration_method = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=100000, confidence=0.999)

# Perform global registration
result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_cloud, target_cloud, source_fpfh, target_fpfh,
    mutual_filter=True,
    max_correspondence_distance=0.01,
    estimation_method=registration_method,
    ransac_n=3,
    criteria=criteria)

# Obtain transformation matrix and apply it to the source point cloud
transformation_matrix = result.transformation
registered_source_cloud = source_cloud.transform(transformation_matrix)

print(transformation_matrix)
o3d.visualization.draw_geometries([registered_source_cloud, target_cloud])