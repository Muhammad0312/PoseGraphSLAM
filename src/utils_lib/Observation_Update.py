import numpy as np

def PoseInversion(Pose):
    '''
    Pose = [x, y, theta]
    '''
    Coordinates = np.array([Pose[0], Pose[1]])
    InverseRotation = np.array([[np.cos(Pose[-1]),  np.sin(Pose[-1])],
                                [-np.sin(Pose[-1]), np.cos(Pose[-1])]])
    InvertedCoordinates = -InverseRotation @ Coordinates

    InvertedPose = np.array([InvertedCoordinates[0], InvertedCoordinates[1], -Pose[-1]])
    return InvertedPose

def PoseInversionJacobian(Pose):
    '''
    Pose = [x, y, theta]
    '''
    x = Pose[0]
    y = Pose[1]
    theta = Pose[2]
    J = np.array([[-np.cos(theta), -np.sin(theta), x*np.sin(theta)-y*np.cos(theta)],
                  [np.sin(theta),  -np.cos(theta), x*np.cos(theta)+y*np.sin(theta)],
                  [       0,               0,               -1                    ]])
    return J

def PoseCompoundingJacobian1(Pose1, Pose2):
    '''
    Pose1 = [x, y, theta] = nXb
    Pose2 = [x, y, theta] = bXc
    '''
    # print("Pose1",Pose1,"Pose2",Pose2)
    theta1 = Pose1[2]
    x2 = Pose2[0]
    y2 = Pose2[1]

    J = np.array([[1, 0, -x2*np.sin(theta1)-y2*np.cos(theta1)],
                  [0, 1,  x2*np.cos(theta1)-y2*np.sin(theta1)],
                  [0, 0,                  1                  ]])
    
    return J

def PoseCompoundingJacobian2(Pose1, Pose2):
    '''
    Pose1 = [x, y, theta] = nXb
    Pose2 = [x, y, theta] = bXc
    '''
    theta1 = Pose1[2]

    J = np.array([[np.cos(theta1), -np.sin(theta1), 0],
                  [np.sin(theta1),  np.cos(theta1), 0],
                  [      0,              0,         1]])
    
    return J

def PoseCompounding(PreviousViewpointInvertedPose, NewViewpointPose):

    x1 = PreviousViewpointInvertedPose[0]
    y1 = PreviousViewpointInvertedPose[1]
    theta1 = PreviousViewpointInvertedPose[2]

    x2 = NewViewpointPose[0]
    y2 = NewViewpointPose[1]
    theta2 = NewViewpointPose[2]

    x = x1 + x2*np.cos(theta1) - y2*np.sin(theta1)
    y = y1 + x2*np.sin(theta1) + y2*np.cos(theta1)
    theta = theta1 + theta2
    CompoundedPose = np.array([x, y, theta])

    return CompoundedPose

def ObservationMatrix(Hp, StateVector, Zp, Rp):
    '''
    StateVector = [x0, y0, theta0, x1, y1, theta1, ..........]
    Hp = [0, 1, 3]
    Zp = [del_x0, del_y0, del_theta0, del_x1, del_y1, del_theta1, .........]
    Rp = []
    '''
    StateVector=np.array(StateVector)
    Hp = np.array(Hp)
    Zp = np.array(Zp)
    Rp = np.array([[0.1, 0, 0],
                [0, 0.1, 0],
                [0, 0, 0.1]])

    Zk = Zp
    Rk = Rp
    Vk = np.eye((Zp.shape)[0])
    Hk = np.zeros((Zp.shape[0], StateVector.shape[0]))

    CurrentViewpoint = StateVector[-3:]
    CurrentViewpointInverted = PoseInversion(CurrentViewpoint)
    r = 0
    for index in Hp:
        # OverlappedViewpoint = StateVector[index*3: index+3]  
        ####### extracting the overlapping isnot correct OverlappedViewpoint [0.10475148]
        OverlappedViewpoint = StateVector[index*3: index*3+3]
        # print("OverlappedViewpoint",OverlappedViewpoint)
        J2 = PoseCompoundingJacobian2(CurrentViewpointInverted, OverlappedViewpoint)
        J1 = PoseCompoundingJacobian1(CurrentViewpointInverted, OverlappedViewpoint)
        J = PoseInversionJacobian(CurrentViewpoint)
        J1J = J1 @ J
        Hk[r:r+3, index:index+3] = J2
        Hk[r:r+3, -6:-3] = J1J 
        Hk[r:r+3, -3: ] = J1J

        r = r + 3
        
    return Zk, Rk, Hk, Vk

#================================================
#===========Seriously Under Construction=========
#================================================

def Update(nXk, nPk, Zk, Rk, Hk, Vk, Hp=0):
    # print("nPk",nPk)

    K_k = nPk @ Hk.T @ np.linalg.inv(Hk @ nPk @ Hk.T + Vk @ Rk @ Vk.T)
    X_k = nXk + K_k @ (Zk - Hk @ nXk)
    P_k = (np.eye(nPk.shape[0]) - K_k @ Hk) @ nPk
    
    return X_k, P_k

state_vector = np.array([0, 0, 0, 1, 1, np.pi/4, 1, 1, np.pi/4])
nPk = np.eye(9)
Hp = np.array([0])
Zp = np.array([1, 1, np.pi/4])
Rp = np.array([[0.1, 0, 0],
               [0, 0.1, 0],
               [0, 0, 0.1]])

Zk, Rk, Hk, Vk = ObservationMatrix(Hp, state_vector, Zp, Rp)
Xk, Pk = Update(state_vector, nPk, Zk, Rk, Hk, Vk)

print('Zk : ', Zk.shape)
print('Rk : ', Rk.shape)
print('Hk : ', Hk.shape)
print('Vk : ', Vk.shape)
print('Xk : ', Xk.shape)
print('Pk : ', Pk.shape)