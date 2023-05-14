import numpy as np
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import sin, cos, atan2

def euclidean_distance(x, y):
  x,y = np.array(x), np.array(y)
  return np.sqrt(np.sum((x - y)**2))

########################################################################

def check_distance_bw_scans(xk, dist_th, ang_th):

    last_scan_pose = xk[-6:-3]  # 2nd last state in state vector
    curr_pose = xk[-3:]         # last state in state vector
    
    dist_since_last_scan = euclidean_distance(last_scan_pose[:2], curr_pose[:2]) 
    rot_since_last_scan = abs(last_scan_pose[2] - curr_pose[2])

    # only add pose/scan if we have move significantly
    if dist_since_last_scan > dist_th:# or rot_since_last_scan > ang_th:
         return True
    else:
         return False
    
########################################################################

def pose_inversion(xy_state):
    x,y,theta = xy_state
    # theta = -theta
    new_x = -x*cos(theta) - y*sin(theta)
    new_y = x*sin(theta) - y*cos(theta)
    new_theta = -theta
    return [new_x,new_y,new_theta] 

########################################################################

def compounding(a_x_b, b_x_c):
    x_b,y_b,theta_b = a_x_b
    x_c,y_c,theta_c = b_x_c

    new_x = x_b + x_c*cos(theta_b) - y_c*sin(theta_b)
    new_y = y_b + x_c*sin(theta_b) + y_c*cos(theta_b)
    new_theta = theta_b + theta_c
    
    return [new_x,new_y,new_theta] 

########################################################################

def get_h(prev_pose, new_pose):
        return compounding(pose_inversion(prev_pose), new_pose)

########################################################################

def angle_wrap(ang):
    """
    Return the angle normalized between [-pi, pi].
    Works with numbers and numpy arrays.
    :param ang: the input angle/s.
    :type ang: float, numpy.ndarray
    :returns: angle normalized between [-pi, pi].
    :rtype: float, numpy.ndarray
    """
    ang = ang % (2 * np.pi)
    if (isinstance(ang, int) or isinstance(ang, float)) and (ang > np.pi):
        ang -= 2 * np.pi
    elif isinstance(ang, np.ndarray):
        ang[ang > np.pi] -= 2 * np.pi
    return ang

########################################################################

def comp(a, b):
    """
    Compose matrices a and b.

    b is the matrix that has to be transformed into a space. Usually used to add the vehicle odometry
    b = [x' y' theta'] in the vehicle frame, to the vehicle position
    a = [x y theta] in the world frame, returning world frame coordinates.

    :param numpy.ndarray a: [x y theta] in the world frame
    :param numpy.ndarray b: [x y theta] in the vehicle frame
    :returns: the composed matrix a+b
    :rtype: numpy.ndarray
    """
    c1 = math.cos(a[2]) * b[0] - math.sin(a[2]) * b[1] + a[0]
    c2 = math.sin(a[2]) * b[0] + math.cos(a[2]) * b[1] + a[1]
    c3 = a[2] + b[2]
    c3 = angle_wrap(c3)
    C = np.array([c1, c2, c3])
    return C

########################################################################

def state_inv(x):
    """
    Inverse of a state vector.

    The world origin as seen in the vehicle frame.

    :param numpy.ndarray x: the state vector.
    :returns: inverse state vector.
    :rtype: numpy.ndarray
    """
    th = angle_wrap(-x[2])
    sinth = math.sin(th)
    costh = math.cos(th)
    dx = costh*(-x[0]) - sinth * (-x[1])
    dy = sinth*(-x[0]) + costh * (-x[1])
    return np.array([dx, dy, th])

########################################################################
def state_inv_jacobian(x):
    """
    Jacobian of the inverse of a state vector.

    The world origin as seen in the vehicle frame Jacobian.

    :param numpy.ndarray x: the state vector.
    :returns: jacobian of inverse state vector.
    :rtype: :py:obj:`numpy.ndarray`
    """
    th = angle_wrap(-x[2])
    sth = math.sin(th)
    cth = math.cos(th)

    J = -np.eye(3)
    J[0, 0] = -cth
    J[0, 1] = sth
    J[0, 2] = -x[0]*sth - x[1]*cth
    J[1, 0] = -sth
    J[1, 1] = -cth
    J[1, 2] = x[0]*cth - x[1]*sth
    return J
    
########################################################################
def compInv(x):
    return state_inv(x),state_inv_jacobian(x)

########################################################################

def yaw_from_quaternion(quat):
    """
    Extract yaw from a geometry_msgs.msg.Quaternion.

    :param geometry_msgs.msg.Quaternion quat: the quaternion.
    :returns: yaw angle from the quaternion.
    :rtype: :py:obj:`float`
    """
    return euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])[2]

########################################################################
def polar2y(line, x):
    """
    Compute the value of y in a line given x.

    Given a line in polar coordinates and the x value of a point computes
    its y value.

    :param numpy.ndarray line: the line as [rho, theta].
    :param float x: the value in x coordinates.
    :returns: the value in y coordinates.
    :rtype: :py:obj:`float`
    """
    sin = np.sin(line[1])
    cos = np.cos(line[1])
    x0 = line[0] * cos
    y0 = line[0] * sin
    m = -cos/sin
    return m*(x-x0) + y0
########################################################################
def ekf_predict(xk,uk,Pk,Qk):
        """
        Implement the prediction equations of the EKF.
        Saves the new robot state and uncertainty.
        Input:
            uk : numpy array [shape (3,) with (dx, dy, dtheta)]
            Qk(3x3)
            xk(zeros(3))
            pk(eye(3))
        """
        # Predict the mean of state vector
        Ak = np.array([[1,0, - np.sin(xk[2]) * uk[0] - np.cos(xk[2]) * uk[1]]
                      ,[0,1,np.cos(xk[2]) * uk[0] - np.sin(xk[2]) * uk[1]]
                      ,[0,0,1]]) 
        Wk = np.array([[np.cos(xk[2]), - np.sin(xk[2]),0],
                       [np.sin(xk[2]),np.cos(xk[2]),0],
                      [0,0,1]])
                # ref: slide (60)- MR4-Gaussian-Filters-KF-EKF-updated.pdf
        xk = comp(xk,uk)  # eq(5.45) in  lecture notes
        Pk = np.dot(np.dot(Ak,Pk),np.transpose(Ak))  + np.dot(np.dot(Wk,Qk),np.transpose(Wk))
########################################################################
def jacobianH(lineworld):
        """
        Compute the jacobian of the get_polar_line function.
        It does it with respect to the robot state xk (done in pre-lab).
        # Complete the Jacobian H from the pre-lab
        # Jacobian H
        # lineworld is [rho_w, phi_w]
        """
        phi_w = lineworld[1]
        d1 = -np.cos(phi_w)
        d2 = -np.sin(phi_w)
        H = np.array([[d1,d2,0],[0,0,-1]])         # Jacobian matrix H 
        return H

########################################################################

def Mahalanobis_distance(v,S):
    """
    takes v (z - h)
    and S(np.dot(np.dot(H,self.Pk),np.transpose(H)) + self.Rk)
    return the Mahalanobis_distance 
    """
    return np.dot(np.dot(np.transpose(v), np.linalg.inv(S)), v)

########################################################################

def ekf_data_association(xk,map,lines,Pk,Rk):
        """
        takes map lines and measured lines 
        nx4 matrix with a segment in each row as [x1 y1 x2 y2]
        Look for closest correspondences.
        The correspondences are between the provided measured lines and the map
        known a priori.
        Return:
          Hk_list : list of 2x3 matrices (jacobian)
          Yk_list : list of 2x1 matrices (innovation)
          Sk_list : list of 2x2 matrices (innovation uncertainty)
          Rk_list : list of 2x2 matrices (measurement uncertainty)
        """
        chi_thres = 0.103   # the larger the more the model accepts more observations and updates more frequently 
        associd = list()
        Hk_list = list()
        Vk_list = list()
        Sk_list = list()
        Rk_list = list()

        # the lengths for map lines and measured lines  (equations from particle filter lab)
        map_line = np.sqrt(np.power(map[:,2] -  map[:,0],2) + np.power(map[:,3] - map[:,1],2))  # map lines (2)
        observed_line = np.sqrt(np.power(lines[:,2] - lines[:,0],2) + np.power(lines[:,3] - lines[:,1],2)) # sensed lines (2)

        # For each observed line
        for i in range(0, lines.shape[0]):
            # Get the polar line representation in robot frame
            z = get_polar_line(lines[i],[0,0,0])
            # Variables for finding minimum
            minD = 1e9
            minj = -1

            # For each line in the known map
            for j in range(0, map.shape[0]):

                # Compute matrices
                h = get_polar_line(map[j],[0,0,0]) # get_polar_line(-,odom) - world coo - calc jacobian 
                H = jacobianH(h,xk)
                h = get_polar_line(map[j],xk) # Map line is in the robot- calc innovation
                v = z - h
                S = np.dot(np.dot(H,Pk),np.transpose(H)) + Rk
                D = Mahalanobis_distance(v,S)

                # check if observed line is longer than map 
                islonger = False
                if observed_line[i] < map_line[j]:
                  islonger = False
                  pass
                else:
                    islonger = True

                # Check if the obseved line is the one with smallest mahalanobis distance
                if np.sqrt(D) < minD and not islonger:
                    minj = j
                    minz = z
                    minh = h
                    minH = H
                    minv = v
                    minS = S
                    minD = D

            # Minimum distance below threshold
            if minD < chi_thres:
                associd.append([i, minj])
                Hk_list.append(minH)
                Vk_list.append(minv)
                Sk_list.append(minS)
                Rk_list.append(Rk)
        return Hk_list, Vk_list, Sk_list, Rk_list

########################################################################

def ekf_update(xk,Pk,Hk_list, Vk_list, Sk_list, Rk_list):
        """
        Update the position of the robot according to the given matrices.
        The matrices contain the current position and the data association
        parameters. All input lists have the same lenght.
        Input:
          Hk_list : list of 2x3 matrices (jacobian)
          Yk_list : list of 2x1 matrices (innovation)
          Sk_list : list of 2x2 matrices (innovation uncertainty)
          Rk_list : list of 2x2 matrices (measurement uncertainty)
        """
        # Compose list of matrices as single matrices
        n = len(Hk_list)
        H = np.zeros((2*n, 3))
        v = np.zeros((2*n))
        S = np.zeros((2*n, 2*n))
        R = np.zeros((2*n, 2*n))

        for i in range(n):
            H[2*i:2*i+2, :] = Hk_list[i]
            v[2*i:2*i+2] = Vk_list[i]
            S[2*i:2*i+2, 2*i:2*i+2] = Sk_list[i]
            R[2*i:2*i+2, 2*i:2*i+2] = Rk_list[i]
        # There is data to update
        if not n > 0:
          #print ("There is data to update")
            return
        # near optimal kalman gain  (substitute by the value of resigual coveriance)
        K = np.dot(np.dot(Pk, np.transpose(H)), np.linalg.inv(S))  # equation (1)
        xk += np.dot(K,v)# Update the mean of state vector  # equation (2)
        I = np.eye(3)
        # Update the uncertainty matrix
        Pk = np.dot(np.dot((I - np.dot(K,H)), Pk),np.transpose(I - np.dot(K,H))) # equation (3)  
