import numpy as np 
import matplotlib.pyplot as plt  
import copy
from scipy.spatial import cKDTree
import data_trial_scans
from math import atan2
from numpy.linalg import inv

def get_displacement_from_T(T):
    x = T[0,2]
    y = T[1,2]
    theta = atan2(T[1,0],T[0,0])
    return [x,y,theta]

def get_transformation_matrix(pose1,pose2):
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
    # print("initial Guess",T)
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

def icp(A, B, pose1, pose2, init_transform=None, max_iterations=20, tolerance=1e-3):
    
    if init_transform is None: 
        init_transform=get_transformation_matrix(pose1, pose2)
        print('init icp transform: ',get_displacement_from_T(init_transform))
    else:
        x, y, theta = init_transform
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        init_transform = np.array([[cos_theta, -sin_theta, x],
                            [sin_theta, cos_theta, y],
                            [0, 0, 1]])
        print('init icp transform: ',get_displacement_from_T(init_transform))
    

    A = np.array([list(x) for x in zip(*A)])
    B = np.array([list(x) for x in zip(*B)])

    pose1 = np.array(pose1)
    pose2 = np.array(pose2)

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
        # Update transformation
        transform = T_optimal
        A_transformed = transform @ A_transformed
        errors[i] = np.sum(distances ** 2)
        single_error=np.sqrt(errors)
        if i > 0 and np.abs(errors[i] - errors[i-1]) < tolerance:
            break
    final_transform = dummy

    displacement = get_displacement_from_T(final_transform)

    return displacement, single_error[:i+1]


