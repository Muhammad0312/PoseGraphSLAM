import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial import KDTree

length = 10  # Length of the rectangular room
width = 6   # Width of the rectangular room
n = 50      # Number of points along the length
m = 50      # Number of points along the width
sigma = 0.05 # Standard deviation of the noise

# Generate the x, y coordinates of the point cloud 1
x = np.zeros((n*2 + m*2 - 4,))
y = np.zeros((n*2 + m*2 - 4,))
idx = 0
for i in range(n):
    x[idx] = (i - (n/2)) * (length/n)
    y[idx] = -width/2
    idx += 1
for i in range(1, m-1):
    x[idx] = length/2
    y[idx] = (i - (m/2)) * (width/m)
    idx += 1
for i in range(n):
    x[idx] = (n/2 - i) * (length/n)
    y[idx] = width/2
    idx += 1
for i in range(1, m-1):
    x[idx] = -length/2
    y[idx] = (m/2 - i) * (width/m)
    idx += 1

# Add noise to the x, y coordinates
x += np.random.normal(loc=0, scale=sigma, size=(n*2 + m*2 - 4,))
y += np.random.normal(loc=0, scale=sigma, size=(n*2 + m*2 - 4,))
z = 0.0*np.ones(len(x))


length1 = 10  # Length of the rectangular room
width1 = 6   # Width of the rectangular room
# Generate the x, y coordinates of the point cloud 2
x1 = np.zeros((n*2 + m*2 - 4,))
y1 = np.zeros((n*2 + m*2 - 4,))
idx = 0
for i in range(n):
    x1[idx] = (i - (n/2)) * (length1/n)
    y1[idx] = -width1/2
    idx += 1
for i in range(1, m-1):
    x1[idx] = length1/2
    y1[idx] = (i - (m/2)) * (width1/m)
    idx += 1
for i in range(n):
    x1[idx] = (n/2 - i) * (length1/n)
    y1[idx] = width1/2
    idx += 1
for i in range(1, m-1):
    x1[idx] = -length1/2
    y1[idx] = (m/2 - i) * (width1/m)
    idx += 1

# Add noise to the x, y coordinates
x1 += np.random.normal(loc=0, scale=sigma, size=(n*2 + m*2 - 4,))
y1 += np.random.normal(loc=0, scale=sigma, size=(n*2 + m*2 - 4,))
z1 = 0.0*np.ones(len(x))

# # Define the plane equation in 3D space (XY plane in this example)
# a, b, c, d = 0, 0, 1, 0
# # Calculate the projection of each point onto the XY plane
# proj_x = np.zeros_like(x)
# proj_y = np.zeros_like(y)
# proj_z = np.zeros_like(z)
# for i in range(n):
#     dist = (a*x[i] + b*y[i] + c*z[i] + d) / np.sqrt(a**2 + b**2 + c**2)
#     proj_x[i] = x[i] - dist*a
#     proj_y[i] = y[i] - dist*b
#     proj_z[i] = z[i] - dist*c

# #====================================================
# scan1 = np.array([[0.0, 0], [0.0, 1], [0.0, 2], [0.0, 3], [1.0, 3], [2.0, 3], [2.5, 2], [3.0, 1], [3.5, 0]])
# scan2 = np.array([[-0.2, 0.2], [-0.06, 1.19], [0.08, 2.18], [0.22, 3.17], [1.21, 3.03], [2.2, 2.89], [2.55, 1.83], [2.91, 0.77], [3.27, -0.29]])

# x = scan1[:,0]
# y = scan1[:,1]
# z = np.zeros(len(x))
# x1 = scan2[:,0]
# y1 = scan2[:,1]
# z1 = np.zeros(len(x1))

point_cloud_1 = []
point_cloud_2 = []
for i in range(0, len(x)):
    point_cloud_1.append([x[i], y[i]])
    point_cloud_2.append([x1[i], y1[i]])

point_cloud_1 = np.array(point_cloud_1)
point_cloud_2 = np.array(point_cloud_2)



# Define a tolerance distance
tolerance = 0.1

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

# Print the overlapping points
print("Overlapping points in cloud 1:", (len(overlapping_points_1)/len(x))*100, '%')
print("Overlapping points in cloud 2:", (len(overlapping_points_2)/len(x1))*100, '%')

#====================================================
# Plot the point clouds
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(projection='3d')
ax1.scatter(x, y, c='blue')
ax1.scatter(x1, y1, c='red')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Original Point Cloud')
# ax2 = fig.add_subplot(122)
# ax2.scatter(proj_x, proj_y, c=proj_z, cmap='jet')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_title('Projected Point Cloud')
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d

# # Define the plane equation in 3D space (XY plane in this example)
# a, b, c, d = 0, 0, 1, 0

# # Generate a random point cloud
# n = 100
# x = np.random.rand(n) * 10 - 5
# y = np.random.rand(n) * 10 - 5
# z = np.random.rand(n) * 10

# # Calculate the projection of each point onto the XY plane
# proj_x = np.zeros_like(x)
# proj_y = np.zeros_like(y)
# proj_z = np.zeros_like(z)
# for i in range(n):
#     dist = (a*x[i] + b*y[i] + c*z[i] + d) / np.sqrt(a**2 + b**2 + c**2)
#     proj_x[i] = x[i] - dist*a
#     proj_y[i] = y[i] - dist*b
#     proj_z[i] = z[i] - dist*c

# # Plot the original and projected point clouds
# fig = plt.figure(figsize=(10, 5))
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(x, y, z, c=z, cmap='jet')
# ax1.set_xlabel('X')
# ax1.set_ylabel('Y')
# ax1.set_zlabel('Z')
# ax1.set_title('Original Point Cloud')
# ax2 = fig.add_subplot(122)
# ax2.scatter(proj_x, proj_y, c=proj_z, cmap='jet')
# ax2.set_xlabel('X')
# ax2.set_ylabel('Y')
# ax2.set_title('Projected Point Cloud')
# plt.show()
