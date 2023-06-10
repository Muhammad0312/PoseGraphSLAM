#!/usr/bin/python3
import numpy as np
import rospy
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

        self.mutex = threading.Lock()

        # robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230

        # Store the scans in the robot's frame
        self.scans = []

        # Store the scans in the world frame using the predicted pose
        self.map = []  # = [s1, s2, s3, s4]

        # Store scan as soon as it is available
        self.scan = []

        # Store groundtruth pose
        self.gt_pose = np.zeros(3)
        # Subscriber to groundtruth

        # Pose initialization
        self.xk = np.array([0, 0.5, 0.0])
        # Edit the initial position of the robot
        self.xk[0] = 3.0
        self.xk[1] = -0.78
        self.xk[2] = np.pi/2.0

        # Groundtruth state vector
        self.gt_xk = np.array([0, 0.5, 0.0])
        self.gt_xk[0] = 3.0
        self.gt_xk[1] = -0.78
        self.gt_xk[2] = np.pi/2.0

        # initial covariance matrix
        self.Pk = np.array([[0.0, 0, 0],    
                            [0, 0.0, 0],
                            [0, 0, 0.0]])   
        
        # Subscriber to lidar
        self.scan_sub = rospy.Subscriber("/kobuki/sensors/rplidar", LaserScan, self.scan_available)

        # Add new pose to keep predicting
        self.xk, self.Pk = AddNewPose(self.xk, self.Pk)

        self.tf_br = TransformBroadcaster()
        # Subscriber to get joint states
        self.js_sub = rospy.Subscriber("/kobuki/joint_states", JointState, self.predict)
        # Odometry noise covariance
        self.Qk = np.array([[0.04, 0],     
                             [0, 0.04]])
        
        self.control_num = 0
        

        self.compass_Vk = np.diag([0.0001])
        
        # define the covariance matrix of the compass
        self.compass_Rk = np.diag([0.157]) 

        # prediction related variables
        self.v=0.0
        self.w=0.0
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_received = False

        # scan related variables
        self.dist_th = 0.2 #0.5   # take scan if displacement is > 0.2m, 0.5
        self.ang_th = 0.1 #0.05 # take scan if angle change is > 0.175(10 degrees), 0.785 (45 degrees)

        self.last_time = rospy.Time.now()

        # If using turtlebot_hoi<>.launch _________________
        self.child_frame_id = "kobuki/base_footprint"
        self.wheel_name_left = "kobuki/wheel_left_joint"
        self.wheel_name_right = "kobuki/wheel_right_joint"

        # odom publisher
        self.odom_pub = rospy.Publisher("kobuki/odom", Odometry, queue_size=10)
        
        # Viewpoints visualizer
        self.viewpoints_pub = rospy.Publisher("/slam/vis_viewpoints",MarkerArray,queue_size=1)
        self.full_map_pub = rospy.Publisher('/slam/map', PointCloud2, queue_size=10)

        # Subscriber for imu data
        self.subImu = rospy.Subscriber('/kobuki/sensors/imu', Imu, self.imu_callback)  

    def wrap_angle(self, angle):
        """this function wraps the angle between -pi and pi

        :param angle: the angle to be wrapped
        :type angle: float

        :return: the wrapped angle
        :rtype: float
        """
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    
    #_______________________   Predictions __________________________________________________________________

    def State_model (self,msg):

        if msg.name[0] == self.wheel_name_left:
            self.left_wheel_velocity = msg.velocity[0]
            self.left_wheel_received = True
            return
        
        elif msg.name[0] == self.wheel_name_right:
            self.right_wheel_velocity = msg.velocity[0]

            if self.left_wheel_received:
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

                self.left_wheel_received = False
    
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
        self.mutex.acquire()
        self.State_model(msg) 
        self.publish_odom_predict(msg)
        self.mutex.release()

    #______________________    Update  ________________________________________________________________

    def imu_callback(self, msg):
        """ imu_callback is a callback function that is called when a new IMU message comes in

        :param msg: the IMU message
        :type msg: Imu

        :return: None
        :rtype: None
        """
        # with self.mutex
        self.mutex.acquire()

        # convert the orientation message received from quaternion to euler
        quaternion = (msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)

        # convert the orientation message received from quaternion to euler
        _, _ , yaw_measurement = euler_from_quaternion(quaternion)

        self.update_heading(yaw_measurement)

        #release the mutex
        self.mutex.release()

    def update_heading(self, yaw_measurement):
        # Create a row vector of zeros of size 1 x 3*num_poses
        Hk = np.zeros((1, len(self.xk)))

        # Replace the last element of the row vector with 1
        Hk[0, -1] = 1

        predicted_compass_meas = self.xk[-1]

        # Compute the kalman gain
        K = self.Pk @ Hk.T @ np.linalg.inv((Hk @ self.Pk @ Hk.T) + (self.compass_Vk @ self.compass_Rk @ self.compass_Vk.T))

        # Compute the innovation
        innovation = self.wrap_angle(yaw_measurement - predicted_compass_meas)

        x = np.dot(K, innovation)

        # Reshape x to (n,)
        x = x.reshape(-1)

        # Update the state vector
        self.xk = self.xk + x

        # Create the identity matrix        
        I = np.eye(len(self.xk))

        # Update the covariance matrix
        self.Pk = (I - K @ Hk) @ self.Pk @ (I - K @ Hk).T
    
    def scan_available(self,scan_msg):

        # Unsubscribe from the joint_state topic and imu topic
        self.subImu.unregister()
        self.js_sub.unregister()
        self.mutex.acquire()

        self.scan = get_scan(scan_msg)

        if len(self.map) == 0:
            # Store the first scan at the robot's first position
            # Store the scan in the robot frame
            self.scans.append(self.scan)

            # Store the scan in the world frame
            self.scan = ScanToWorldFrame(self.xk[0:3], self.scan)
            self.map.append(self.scan)
        
        if check_distance_bw_scans(self.xk, self.dist_th, self.ang_th):  
            print('Entering Update')  

            # Save new scan in the robot frame
            self.scans.append(self.scan)

            # Save new scan in the world frame
            self.scan = ScanToWorldFrame(self.xk[-3:], self.scan)
            self.map.append(self.scan)
            
            # Add new pose to keep predicting
            self.xk, self.Pk = AddNewPose(self.xk, self.Pk)

            # Store the actual viewpoint in the groundtruth state vector
            # self.gt_xk = np.hstack((self.gt_xk, self.gt_pose))

            # Overlapping Scans
            offset = 3
            Ho = OverlappingScans(self.xk, self.map, offset)
            print('Overlap Ho: ', Ho)
            
            # for each matched pair
            for j in Ho:
                
                # Get the scan which overlaps with the lastest scan
                match_scan = self.map[j]

                # Get the viewpoint for the latest scan
                curr_viewpoint = self.xk[-3:]

                # Get the viewpoint for the matched scan
                matched_viewpoint = self.xk[j*3: 3*j+3]

                # curr_viewpoint_gt = self.gt_xk[-3: ]
                # matched_viewpoint_gt = self.gt_xk[j*3: 3*j+3]

                # Obervation Model
                guess_displacement = get_h(curr_viewpoint, matched_viewpoint)

                # actual_displacement = get_h(curr_viewpoint_gt, matched_viewpoint_gt)

                # Get the transformation between two scans using ICP
                zk = ICP(match_scan, self.map[-1], guess_displacement)

                #call new_observationHk
                Hk = self.new_observationHk(j, curr_viewpoint, matched_viewpoint)

                Rk = np.eye(3) * 0.2
                Vk = np.eye(3) 

                innovation = zk - guess_displacement
                

                S = (Hk @ self.Pk @ Hk.T) + (Vk @ Rk @ Vk.T)

                # compute the mahalanobis distance D
                D = innovation.T @ np.linalg.inv(S) @ innovation

                #check if the mahalanobis distance is within the threshold
                if np.sqrt(D) <= 0.8:
                    print('updated')
                    self.update_slam(zk,Hk, Rk, Vk, guess_displacement)
                
            # Publish the viewpoints for the gathered scans
            self.publish_viewpoints()

            # Publish all the scans using the updated state vector
            self.publish_full_map()
        
        self.mutex.release()

        # Subscribe to the joint_state topic and imu topic
        self.js_sub = rospy.Subscriber("/kobuki/joint_states", JointState, self.predict)
        self.subImu = rospy.Subscriber('/kobuki/sensors/imu', Imu, self.imu_callback)
            

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
        # Hk[:, -6:-3] = j1_jominus

        #replace the columns [:, 3*scan_index: 3*scan_index+3] with j2_plus
        Hk[:, 3*scan_index: 3*scan_index+3] = j2_plus

        return Hk



    def update_slam(self, zk, Hk, Rk, Vk, displacement_guess):
        """ This is the update step of the EKF SLAM algorithm

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

    def get_gt(self, msg):
        euler = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.gt_pose[0] = msg.pose.pose.position.x
        self.gt_pose[1] = msg.pose.pose.position.y
        self.gt_pose[2] = euler[-1]

    ##################      Publishing   ##############################

    def publish_full_map(self):
        full_map = scans_to_map(self.xk, self.scans)

        # Create the header for the point cloud message
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world_ned'  # Set the frame ID

        # Create the point cloud message
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, full_map)

        self.full_map_pub.publish(point_cloud_msg)

    def publish_viewpoints(self):

        marker_frontier_lines = MarkerArray()
        marker_frontier_lines.markers = []

        viewpoints_list = []

        for i in range(0,len(self.xk),3):
            myMarker = Marker()
            myMarker.header.frame_id = "world_ned"
            myMarker.type = myMarker.SPHERE
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
        odom.child_frame_id = self.child_frame_id

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

if __name__ == '__main__':

    rospy.init_node("PoseGraphSLAM")

    robot = PoseGraphSLAM()

    rospy.spin()
