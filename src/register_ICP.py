import numpy as np 
import matplotlib.pyplot as plt  
import copy
from scipy.spatial import cKDTree
import data_trial_scans

def get_transformation_matrix():
    pose1 = [8.952061389220677316e-07, -6.361812210453066930e-11, -1.044700073057142620e-05]
    pose2 = [2.883825026077073139e-01, -1.402247462826852198e-01, -9.072827003143387747e-01]
    x1, y1, theta1 = pose1
    x2, y2, theta2 = pose2
    
    cos_theta1 = np.cos(theta1)
    sin_theta1 = np.sin(theta1)
    cos_theta2 = np.cos(theta2)
    sin_theta2 = np.sin(theta2)
    
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    T = np.array([
        [cos_theta1*cos_theta2 - sin_theta1*sin_theta2, -cos_theta1*sin_theta2 - sin_theta1*cos_theta2, delta_x],
        [sin_theta1*cos_theta2 + cos_theta1*sin_theta2, -sin_theta1*sin_theta2 + cos_theta1*cos_theta2, delta_y],
        [0, 0, 1]
    ])
    print("initial Guess",T)
    return np.array(T)

def ls_minimization(X, Y, W):
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)
    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y
    S = X_centered.T @ W @ Y_centered
    U, _, Vt = np.linalg.svd(S)
    R_optimal = Vt.T @ U.T
    t_optimal = centroid_Y.T - R_optimal @ centroid_X.T
    T = np.eye(3)
    T[:2, :2] = R_optimal[:2, :2]
    T[:2, 2] = t_optimal[:2]
    return T

def find_closest_points(A_transformed, B):
    tree = cKDTree(B.T)
    distances, indices = tree.query(A_transformed[:2, :].T)
    return distances, indices


def icp(A, B, init_transform=None, max_iterations=20, tolerance=1e-3):
    if init_transform is None: 
        # init_transform=[[ 0.61588057,  0.78783953,  0.28838161],
        #                 [-0.78783953 , 0.61588057 ,-0.14022475],
        #                 [ 0.          ,0.          ,1.        ]]   
        # init_transform=[[-0.07018052 , 0.99753431 , 0.00588202],[-0.99753431, -0.07018052 ,-0.0070268 ],[ 0.    ,      0.    ,      1.        ]]
        init_transform=get_transformation_matrix()

        
    A_transformed = np.vstack((A, np.ones((1, A.shape[1]))))
    A_transformed = init_transform @ A_transformed[:3, :]
    First_A_transformed= A_transformed
    transform = init_transform
    transform = np.array(transform)
    init=np.array(init_transform)
    errors = np.zeros(max_iterations)
    dummy = init_transform
    dummy = np.array(dummy)
    for i in range(max_iterations):
        distances, indices = find_closest_points(A_transformed[:2, :], B)
        X = A_transformed[:2, :].T
        Y = B[:, indices].T
        W = np.diag(distances ** 2)

        T_optimal = ls_minimization(X, Y, W)
        dummy = T_optimal@dummy
        print ("dummy",)
        # Update transformation
        transform = T_optimal
        A_transformed = transform @ A_transformed
        errors[i] = np.sum(distances ** 2)
        single_error=np.sqrt(errors)
        if i > 0 and np.abs(errors[i] - errors[i-1]) < tolerance:
            break
    # final_transform =transform @init_transform
    final_transform = dummy
    print("Final transform:\n", final_transform)
    return final_transform, single_error[:i+1],First_A_transformed


#------------------------------------------------------------------------------trying the example -------------------------#

A = []
with open('data_trial_scans/scan1.txt', 'r') as f:
    for line in f:
        row = line.strip().split()
        x = float(row[0])
        y = float(row[1])
        A.append([x, y])

A = np.array(A)

B = []
with open('data_trial_scans/scan3.txt', 'r') as f:
    for line in f:
        row = line.strip().split()
        x = float(row[0])
        y = float(row[1])
        B.append([x, y])

B = np.array(B)

x = A[:, 0]
y = A[:, 1]
A= np.vstack((x, y))

x2 = B[:, 0]
y2 = B[:, 1]
B= np.vstack((x2, y2))
# print("B",B)
final_transform, errors ,First_A_transformed= icp(A, B)
A_transformed = final_transform[:2, :2] @ A + final_transform[:2, 2].reshape(-1, 1)


plt.figure(figsize=(10, 10))

# Plot errors
plt.plot(errors, label='Errors')
plt.legend()
plt.show()

# Set figure size
plt.figure(figsize=(10, 10))

# # Plot scatter plots
plt.scatter(A[0, :], A[1, :], label='A')
plt.scatter(B[0, :], B[1, :], label='B')
plt.scatter(A_transformed[0, :], A_transformed[1, :], label='Transformed A')
plt.scatter(First_A_transformed[0, :], First_A_transformed[1, :], label='First_A_transformed')

plt.legend()
plt.show()
