#!/usr/bin/python3

import numpy as np
import rospy, math
from nav_msgs.msg import Odometry 
from std_msgs.msg import Float64, Float32MultiArray
from sensor_msgs.msg import JointState, Imu
from tf.broadcaster import TransformBroadcaster
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import math
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
import math
from utils_lib.helper_functions import *
from utils_lib.add_new_pose import AddNewPose  
from utils_lib.get_scan import get_scan  
from utils_lib.overlapping_scan import OverlappingScans
from utils_lib.register_ICP import icp
from utils_lib.Observation_Update import*
from utils_lib.scans_to_map import scans_to_map

from icp_testing.icp import ICP, ScanToWorldFrame

import sys
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from sensor_msgs import point_cloud2
import threading

import time
import matplotlib.pyplot as plt


class PoseGraphSLAM:
    def __init__(self) -> None:
        # Suppress scientific notations while displaying numbers
        np.set_printoptions(suppress=True)
        self.previous = time.time()

        # self.mutex = threading.Lock()
        # robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230
        self.update_running = False

        self.scans = []

        # Store the map
        self.map = []  # = [s1, s2, s3, s4]
        # Store scan as soon as it is available
        self.scan = []

        # Pose initialization
        self.xk = np.array([0.0, 0.0, 0.0])

        # initial covariance matrix
        self.Pk = np.array([[0.1, 0, 0],    
                            [0, 0.1, 0],
                            [0, 0, 0.1]])   
        
        # Subscriber to lidar
        # self.scan_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/rplidar", LaserScan, self.scan_available)

        # Add new pose to keep predicting
        self.xk, self.Pk = AddNewPose(self.xk, self.Pk)

        self.tf_br = TransformBroadcaster()
        # Subscriber to get joint states
        self.js_sub = rospy.Subscriber("/turtlebot/joint_states", JointState, self.predict)
        # Odometry noise covariance
        self.Qk = np.array([[0.04, 0],     
                             [0, 0.04]])
        
        self.control_num = 0
        

        self.compass_Vk = np.diag([0.0001])
        
        # define the covariance matrix of the compass
        self.compass_Rk = np.diag([0.157]) 

        # prediction related variables
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.v=0.0
        self.w=0.0
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_received = False

        # scan related variables
        self.dist_th = 0.2#05   # take scan if displacement is > 0.2m, 0.5
        self.ang_th = 0.1#0.05 # take scan if angle change is > 0.175(10 degrees), 0.785 (45 degrees)

        self.last_time = rospy.Time.now()

        # If using turtlebot_hoi<>.launch _________________
        self.child_frame_id = "/turtlebot/kobuki/base_footprint"
        self.wheel_name_left = "turtlebot/kobuki/wheel_left_joint"
        self.wheel_name_right = "/turtlebot/kobuki/wheel_right_joint"

        # odom publisher
        self.odom_pub = rospy.Publisher("kobuki/odom", Odometry, queue_size=10)
        
        # Viewpoints visualizer
        self.viewpoints_pub = rospy.Publisher("/slam/vis_viewpoints",MarkerArray,queue_size=1)
        self.full_map_pub = rospy.Publisher('/slam/map', PointCloud2, queue_size=10)

        # create a subscriber to get the scan

        # self.subImu = rospy.Subscriber('/turtlebot/kobuki/sensors/imu_data', Imu, self.imu_callback)

    def wrap_angle(self, angle):
        """this function wraps the angle between -pi and pi

        :param angle: the angle to be wrapped
        :type angle: float

        :return: the wrapped angle
        :rtype: float
        """
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )

    def imu_callback(self, msg):
        """ imu_callback is a callback function that is called when a new IMU message comes in

        :param msg: the IMU message
        :type msg: Imu

        :return: None
        :rtype: None
        """
        # with self.mutex:t
        # self.mutex.acquire()
        # convert the orientation message received from quaternion to euler
        quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

        # convert the orientation message received from quaternion to euler
        _, _ , yaw_measurement = euler_from_quaternion(quaternion)

        #create a row vector of zeros of size 1 x 3*num_poses
        Hk = np.zeros((1, len(self.xk)))

        #replace the last element of the row vector with 1
        Hk[0, -1] = 1

        #multiply the row vector with the state vector to get the observation
        # predicted_compass_meas = Hk @ self.xk + self.compass_Vk

        predicted_compass_meas = self.xk[-1]

        #compute the kalman gain
        K = self.Pk @ Hk.T @ np.linalg.inv((Hk @ self.Pk @ Hk.T) + (self.compass_Vk @ self.compass_Rk @ self.compass_Vk.T))

        #compute the innovation
        innovation = self.wrap_angle(yaw_measurement - predicted_compass_meas)

        x = np.dot(K, innovation)

        #  reshape x to (n,)
        x = x.reshape(-1)

        # update the state vector
        self.xk = self.xk + x

        #create the identity matrix        
        I = np.eye(len(self.xk))

        #update the covariance matrix
        self.Pk = (I - K @ Hk) @ self.Pk @ (I - K @ Hk).T

        # print('heading updated')
        # release the mutex
        # self.mutex.release()

        pass
    #_______________________   Predictions __________________________________________________________________

    def State_model (self,msg):
        if msg.name[0] == self.wheel_name_left:
            self.left_wheel_velocity = msg.velocity[0]
            self.right_wheel_velocity = msg.velocity[1]

            # Do calculations
            left_lin_vel = self.left_wheel_velocity * self.wheel_radius
            right_lin_vel = self.right_wheel_velocity * self.wheel_radius

            self.v = (left_lin_vel + right_lin_vel) / 2.0
            self.w = (left_lin_vel - right_lin_vel) / self.wheel_base_distance
        
            #calculate dt
            current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            dt = (current_time - self.last_time).to_sec()
            self.last_time = current_time

            Ak = np.array([[1, 0, -self.v * dt * np.sin(self.xk[-1])],
                            [0, 1, self.v * dt * np.cos(self.xk[-1])],
                            [0, 0, 1]]) 

            Wk = np.array([[dt*self.wheel_radius*np.cos(self.xk[-1])/2,   dt*self.wheel_radius*np.cos(self.xk[-1])/2],
                            [dt*self.wheel_radius*np.sin(self.xk[-1])/2,  dt*self.wheel_radius*np.sin(self.xk[-1])/2],
                            [dt*self.wheel_radius/self.wheel_base_distance,    -dt*self.wheel_radius/self.wheel_base_distance]])
            
            F1k, F2k = self.get_F1k_F2k(Ak, Wk)

            self.Pk = F1k @ self.Pk @ F1k.T  + F2k @ self.Qk @ F2k.T

            # State updates x' = x + d * cos(theta) y' = y + d * sin(theta)
            self.xk[-3] = self.xk[-3] + self.v * dt * np.cos(self.xk[-1])
            self.xk[-2] = self.xk[-2] + self.v * dt * np.sin(self.xk[-1]) 
            self.xk[-1] = self.xk[-1] + self.w * dt

    
    def get_F1k_F2k(self, Ak, Wk):
        F1k = np.zeros((len(self.xk),len(self.xk)))
        F1k[-3:,-3:] = Ak
        for i in range(len(self.xk)-3):
            F1k[i,i] = 1.0  

        F2k = np.zeros((len(self.xk),2),np.float32)
        F2k[-3:] = Wk

        return F1k, F2k

    def predict (self,msg):
        # Use mutex to prevent different subscriber from using the same resource simultaneously
        # with self.mutex:
        # self.mutex.acquire()
        self.State_model(msg) 
        self.publish_odom_predict(msg)
        # self.mutex.release()

    #______________________    Update  ________________________________________________________________
    

    def scan_available(self,scan_msg):
        #unsubscribe from the scan topic
        # self.subImu.unregister()
        # self.js_sub.unregister()

        self.scan = get_scan(scan_msg)

        if len(self.map) == 0:
            self.scans.append(self.scan)
            self.scan = ScanToWorldFrame(self.xk[0:3], self.scan)
            self.map.append(self.scan)
        
        if check_distance_bw_scans(self.xk, self.dist_th, self.ang_th):
        # if self.control_num == 4:
            # with self.mutex:   
            if True:    
                print('Entering Update')  
                # add new scan
                self.scans.append(self.scan)
                self.scan = ScanToWorldFrame(self.xk[-3:], self.scan)

                self.map.append(self.scan)

                self.xk, self.Pk = AddNewPose(self.xk, self.Pk)

                #print the length of map and the state vector
                # print('map length: ', len(self.map))
                # print('xk length: ', len(self.xk))

                # Overlapping Scans
                offset = 4
                Ho = OverlappingScans(self.xk, self.map, offset)

                '''For debugging purposes'''
                print('Overlap Ho: ', Ho)

                # for each matched pair
                for j in Ho:
                    # print('------------- matching started ---------')
                    # print('scan index: ', j)
                    match_scan = self.map[j]
                    
                    curr_viewpoint = self.xk[-3:]
                    matched_viewpoint = self.xk[j*3: 3*j+3]

                    # Obervation Model
                    guess_displacement = get_h(curr_viewpoint, matched_viewpoint)

                    zk = ICP(match_scan, self.map[-1], guess_displacement)

                    Hk = self.new_observationHk(j, curr_viewpoint, matched_viewpoint)

                    Rk = np.eye(3) * 0.2
                    Vk = np.eye(3) 

                    innovation = zk - guess_displacement

                    S = (Hk @ self.Pk @ Hk.T) + (Vk @ Rk @ Vk.T)

                    # compute the mahalanobis distance D
                    D = innovation.T @ np.linalg.inv(S) @ innovation

                    #check if the mahalanobis distance is within the threshold
                    if np.sqrt(D) <= 0.3: 
                        self.update_slam(zk,Hk, Rk, Vk, guess_displacement)
                    
                print('Exiting Update')
                self.publish_viewpoints()
                self.publish_full_map()

            self.control_num = 0

        else:
            self.control_num += 1

        # self.mutex.release()
        # self.js_sub = rospy.Subscriber("/turtlebot/joint_states", JointState, self.predict)
        # self.subImu = rospy.Subscriber('/turtlebot/kobuki/sensors/imu_data', Imu, self.imu_callback)
            

    def new_observationHk(self, scan_index, current_viewpoint, matched_viewpoint):

        Hk = np.zeros((3, len(self.xk)))


        j2_plus = np.array([[np.cos(current_viewpoint[-1]), np.sin(current_viewpoint[-1]), 0], 
                                    [-np.sin(current_viewpoint[-1]), np.cos(current_viewpoint[-1]), 0],                            
                                    [0, 0, 1]])

        j1_jominus = np.array([[-np.cos(current_viewpoint[-1]), -np.sin(current_viewpoint[-1]), np.sin(current_viewpoint[-1])*(current_viewpoint[0] - matched_viewpoint[0]) + np.cos(current_viewpoint[-1])*(matched_viewpoint[1] - current_viewpoint[1])],
                                [np.sin(current_viewpoint[-1]), -np.cos(current_viewpoint[-1]), np.cos(current_viewpoint[-1])*(current_viewpoint[0] - matched_viewpoint[0]) + np.sin(current_viewpoint[-1])*(current_viewpoint[1] - matched_viewpoint[1])],
                                [0,0,-1]])
        
        #replace the last 3 columns of Hk with j1_jominus
        Hk[:, -3:] = j1_jominus

        #replace columns [:,-6:-3] with j1_jominus#
        Hk[:, -6:-3] = j2_plus

        #replace the columns [:, 3*scan_index: 3*scan_index+3] with j2_plus
        Hk[:, 3*scan_index: 3*scan_index+3] = j2_plus

        return Hk



    def update_slam(self, zk, Hk, Rk, Vk, displacement_guess):
        """ this is the update step of the EKF SLAM algorithm

        :param zk: the observation vector
        :type zk: numpy array
        :param Hk: the observation matrix
        :type Hk: numpy array
        :param Rk: the observation noise covariance matrix
        :type Rk: numpy array
        :param Vk: the observation noise matrix
        :type Vk: numpy array
        :param displacement_guess: the guess of the displacement vector
        :type displacement_guess: numpy array

        :return: None

        """
        innovation = zk - displacement_guess

        K = self.Pk @ Hk.T @ np.linalg.inv((Hk @ self.Pk @ Hk.T) + (Vk @ Rk @Vk.T))

        #update the state vector
        self.xk = self.xk + K @ innovation

        #update the covariance matrix
        self.Pk = (np.eye(len(self.xk)) - K @ Hk) @ self.Pk @ (np.eye(len(self.xk)) - K @ Hk).T
       
        pass

    ##################      Publishing   #############################

    def publish_full_map(self):
        # print('State vector: ',self.xk.shape)
        # print('Map: ', len(self.map))
        full_map = scans_to_map(self.xk, self.scans)
        # print('map_shape: ', full_map)

        # Create the header for the point cloud message
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world_ned'  # Set the frame ID

        # Create the point cloud message
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, full_map)

        self.full_map_pub.publish(point_cloud_msg)

    def check_obs_model(self):
        ref_list = []
        h_list = []

        for i in range(0,len(self.xk)-3,3):
            ref_pose = self.xk[i:i+3]
            next_pose = self.xk[i+3:i+6]
            # x, y, theta
            h = get_h(ref_pose, next_pose)

            ref_list.append(ref_pose)
            h_list.append(h)

    def publish_h_lines(self, ref_list, h_list):

        viewpoints_list = []

        # Loop through the data and create markers
        for i in range(len(ref_list)-1):
            # Create a marker for each line segment
            myMarker = Marker()
            myMarker.header.frame_id = "world_ned"
            myMarker.type = myMarker.LINE_LIST
            myMarker.action = myMarker.ADD
            myMarker.id = i 

            # Set the marker properties
            myMarker.pose.orientation.w = 1.0
            myMarker.scale.x = 0.05

            # Set the line segment endpoints
            startPoint = Point()
            endPoint = Point()

            startPoint.x = ref_list[i][0]
            startPoint.y = ref_list[i][1]
            startPoint.z = 0.0

            endPoint.x = ref_list[i][0] + h_list[i][0]
            endPoint.y = ref_list[i][1] + h_list[i][1]
            endPoint.z = 0.0

            myMarker.points.append(startPoint)
            myMarker.points.append(endPoint)

            myMarker.color = ColorRGBA(1, 1, 0, 1)  # Green color

            viewpoints_list.append(myMarker)

        self.h_lines_pub.publish(viewpoints_list)

    def publish_viewpoints(self):

        marker_frontier_lines = MarkerArray()
        marker_frontier_lines.markers = []

        viewpoints_list = []

        for i in range(0,len(self.xk),3):
            myMarker = Marker()
            myMarker.header.frame_id = "world_ned"
            myMarker.type = myMarker.SPHERE # sphere
            myMarker.action = myMarker.ADD
            myMarker.id = i

            myMarker.pose.orientation.x = 0.0
            myMarker.pose.orientation.y = 0.0
            myMarker.pose.orientation.z = 0.0
            myMarker.pose.orientation.w = 1.0

            myPoint = Point()
            myPoint.x = self.xk[i]
            myPoint.y = self.xk[i+1]

            myMarker.pose.position = myPoint
            
            myMarker.color=ColorRGBA(0.224, 1, 0, 1)
            myMarker.scale.x = 0.1
            myMarker.scale.y = 0.1
            myMarker.scale.z = 0.05
            # self.myMarker.lifetime = rospy.Duration(0)
            viewpoints_list.append(myMarker)

        self.viewpoints_pub.publish(viewpoints_list)

    def publish_odom_predict(self,msg):

        current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        q = quaternion_from_euler(0, 0, self.xk[-1])
        
        covar = [self.Pk[-3,-3], self.Pk[-3,-2], 0.0, 0.0, 0.0, self.Pk[-3,-1],
                self.Pk[-2,-3], self.Pk[-2,-2], 0.0, 0.0, 0.0, self.Pk[-2,-1],  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                self.Pk[-1,-3], self.Pk[-1,-2], 0.0, 0.0, 0.0, self.Pk[-1,-1]]

        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "world_ned"
        odom.child_frame_id = "turtlebot/kobuki/base_footprint"

        odom.pose.pose.position.x = self.xk[-3]
        odom.pose.pose.position.y = self.xk[-2]

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w
        odom.pose.covariance = covar

        self.odom_pub.publish(odom)

        self.tf_br.sendTransform((self.xk[-3], self.xk[-2], 0.0), q, rospy.Time.now(), odom.child_frame_id, odom.header.frame_id)

    #______________________________________________________________________-##########################################

if __name__ == '__main__':

    rospy.init_node("PoseGraphSLAM")

    robot = PoseGraphSLAM()

    rospy.spin()