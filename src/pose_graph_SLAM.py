#!/usr/bin/python3

import numpy as np
import rospy, math
from nav_msgs.msg import Odometry 
from std_msgs.msg import Float64, Float32MultiArray
from sensor_msgs.msg import JointState
from tf.broadcaster import TransformBroadcaster
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from math import sin, cos, atan2
import math
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan

from utils_lib.helper_functions import *
  
class PoseGraphSLAM:
    def __init__(self) -> None:

        # robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230
        
        # Pose initialization
        self.xk = np.zeros(3)

        # initial covariance matrix
        self.Pk = np.array([[0.04, 0, 0],    
                            [0, 0.04, 0],
                            [0, 0, 0.04]])     
        # odometry noise covariance
        self.Qk = np.array([[0.05**2, 0],     
                             [0, 0.05**2]])


        # prediction related variables
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.v=0.0
        self.w=0.0
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_received = False

        # scan related variables
        self.dist_th = 0.2   # take scan if displacement is > 0.2m
        self.ang_th = 0.174533 # take scan if angle change is > 10 degrees

        self.scans = []

        self.last_time = rospy.Time.now()


        # If using kobuki_basic.launch _________________
        # joint state subscriber
        # self.js_sub = rospy.Subscriber("kobuki/joint_states", JointState, self.predict)
        # # scan subscirber
        # self.scan_sub = rospy.Subscriber("/kobuki/sensors/rplidar", LaserScan, self.get_scan)
        # self.child_frame_id = "kobuki/base_footprint"
        # self.wheel_name_left = "kobuki/wheel_left_joint"
        # self.wheel_name_right = "kobuki/wheel_right_joint"

        # If using turtlebot_hoi<>.launch _________________
        self.js_sub = rospy.Subscriber("/turtlebot/joint_states", JointState, self.predict)
        self.odom_pub = rospy.Subscriber("/turtlebot/kobuki/sensors/rplidar", LaserScan, self.get_scan)
        self.child_frame_id = "turtlebot/kobuki/base_footprint"
        self.wheel_name_left = "turtlebot/kobuki/wheel_left_joint"
        self.wheel_name_right = "turtlebot/kobuki/wheel_right_joint"


        # odom publisher
        self.odom_pub = rospy.Publisher("kobuki/odom", Odometry, queue_size=10)
        

        self.tf_br = TransformBroadcaster()
    
    #_______________________   Predictions __________________________________________________________________

    def State_model (self,msg):
        # print(msg.name[0])

        if msg.name[0] == self.wheel_name_left:
            self.left_wheel_velocity = msg.velocity[0]
            print('got left wheel')
            # self.left_wheel_received = True
                # return
        elif msg.name[0] == self.wheel_name_right:
            self.right_wheel_velocity = msg.velocity[0]
            print('got right wheel')
            # self.right_wheel_received = True

        # if (self.left_wheel_received and self.right_wheel_received):
            # Do calculations
        left_lin_vel = self.left_wheel_velocity * self.wheel_radius
        right_lin_vel = self.right_wheel_velocity * self.wheel_radius

        self.v = (left_lin_vel + right_lin_vel) / 2.0
        self.w = (right_lin_vel - left_lin_vel) / self.wheel_base_distance
    
        #calculate dt
        current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time
        
        
        # State updates x' = x + d * cos(theta) y' = y + d * sin(theta)
        self.xk[-3] = self.xk[-3] + self.v * dt * np.cos(self.xk[-1])
        self.xk[-2] = self.xk[-2] + self.v * dt * np.sin(self.xk[-1]) 
        self.xk[-1] = self.xk[-1] + self.w * dt

        return dt 
    
    def get_F1k_F2k(self, Ak, Wk):

        I = np.eye(3).astype(np.float32)
        F1k = np.zeros((len(self.xk),len(self.xk)))
        F1k[-3:,-3:] = Ak
        for i in range(len(self.xk)-3):
            F1k[i,i] = 1.0  

        F2k = np.zeros((len(self.xk),2),np.float32)
        F2k[-3:] = Wk

        return F1k, F2k
     #########################-_________________________________________________________________________________________________-##########################################

    def predict (self,msg):
        # print('in predict')
        dt = self.State_model(msg) 

        Ak = np.array([[1, 0, -self.v * dt * np.sin(self.xk[-1])],
                       [0, 1, self.v * dt * np.cos(self.xk[-1])],
                       [0, 0, 1]]) 

        Wk = np.array([[dt*self.wheel_radius*np.cos(self.xk[-1])/2,   dt*self.wheel_radius*np.cos(self.xk[-1])/2],
                        [dt*self.wheel_radius*np.sin(self.xk[-1])/2,  dt*self.wheel_radius*np.sin(self.xk[-1])/2],
                        [dt*self.wheel_radius/self.wheel_base_distance,    -dt*self.wheel_radius/self.wheel_base_distance]])
            
        # print(self.xk[-1])

        F1k, F2k = self.get_F1k_F2k(Ak, Wk)

        self.Pk = F1k @ self.Pk @ F1k.T  + F2k @ self.Qk @ F2k.T

        # print(self.Pk)

        self.publish_odom_predict(msg)

    #______________________    Update  ________________________________________________________________
    
    def get_scan(self, scan_msg):

        ranges = np.array(scan_msg.ranges)
        angle_min = scan_msg.angle_min
        angle_increment = scan_msg.angle_increment
        
        num_points = len(ranges)
        angles = np.linspace(angle_min, angle_min + angle_increment * (num_points - 1), num_points)
        
        curr_scan = []
        for i in range(num_points):
            if ranges[i] < scan_msg.range_max and ranges[i] > scan_msg.range_min:
                x = ranges[i] * math.cos(angles[i])
                y = ranges[i] * math.sin(angles[i])
                curr_scan.append((x, y))
            
        curr_scan = np.array(curr_scan)

        np.savetxt('pose3.txt', self.xk[-3:])
        np.savetxt('scan3.txt', curr_scan)

        # print('scan len: ', curr_scan.shape[0])
        print('xk: ', self.xk)
        # print('curr_scan: ', curr_scan[-5:])

        if len(self.xk) == 3 : #whatever condition
            # add new scan
            self.scans.append(curr_scan)
            self.add_new_pose()
        else:
            last_scan_pose = self.xk[-6:-3]  # 2nd last state in state vector
            curr_pose = self.xk[-3:]         # last state in state vector
            
            dist_since_last_scan = euclidean_distance(last_scan_pose[:2], curr_pose[:2]) 
            rot_since_last_scan = abs(last_scan_pose[:2] - curr_pose[:2])

            print('dist_since_last_scan: ', dist_since_last_scan)
            print('rot_since_last_scan: ', rot_since_last_scan)

            if dist_since_last_scan > self.dist_th or rot_since_last_scan > self.ang_th:
                # add new scan
                self.scans.append(curr_scan)
                self.add_new_pose()
    

    def add_new_pose(self):
        # print('In add new pose')
        pass

    ##################      Publishing   ##############################

    #########################-_________________________________________________________________________________________________-##########################################
    
    def publish_odom_predict(self,msg):
        current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        q = quaternion_from_euler(0, 0, self.xk[-1])
        # print (self.Pk )
        # N = np.zeros((6,6))
        # for i in range (2):
        #     for j in range (2):
        #         N[i,j] = self.Pk[i,j]

        # N[5] = [0,0,0,self.Pk[2,0],self.Pk[2,1],self.Pk[2,2]] 
        
        covar = [self.Pk[0,0], self.Pk[0,1], 0.0, 0.0, 0.0, self.Pk[0,2],
                self.Pk[1,0], self.Pk[1,1], 0.0, 0.0, 0.0, self.Pk[1,2],  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                self.Pk[2,0], self.Pk[2,1], 0.0, 0.0, 0.0, self.Pk[2,2]]

        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "world"
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


    #______________________________________________________________________-##########################################

if __name__ == '__main__':

    rospy.init_node("PoseGraphSLAM")

    robot = PoseGraphSLAM()

    rospy.spin()

    





