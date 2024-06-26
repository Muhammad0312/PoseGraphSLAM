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
from utils_lib.add_new_pose import AddNewPose  
# from utils_lib.overlapping_scan import OverlappingScans
# from utils_lib.register_ICP import icp

def pose_inversion(xy_state):
    x,y,theta = xy_state
    # theta = -theta
    new_x = -x*cos(theta) - y*sin(theta)
    new_y = x*sin(theta) - y*cos(theta)
    new_theta = -theta
    return [new_x,new_y,new_theta] 

def compounding(a_x_b, b_x_c):
    x_b,y_b,theta_b = a_x_b
    x_c,y_c,theta_c = b_x_c

    new_x = x_b + x_c*cos(theta_b) - y_c*sin(theta_b)
    new_y = y_b + x_c*sin(theta_b) + y_c*cos(theta_b)
    new_theta = theta_b + theta_c
    
    return [new_x,new_y,new_theta] 

class PoseGraphSLAM:
    def __init__(self) -> None:

        # robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230
        self.update_running = False

        # Pose initialization
        self.xk = np.zeros(3)

        # initial covariance matrix
        self.Pk = np.array([[0.04, 0, 0],    
                            [0, 0.04, 0],
                            [0, 0, 0.04]])     
        # odometry noise covariance
        self.Qk = np.array([[0.05**2, 0],     
                             [0, 0.05**2]])

        self.Rk = np.array([[0.05**2, 0],     
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
        self.dist_th = 0.5   # take scan if displacement is > 0.2m, 0.5
        self.ang_th = 0.785 # take scan if angle change is > 0.175(10 degrees), 0.785 (45 degrees) 

        self.scans = []  # = [s1, s2, s3, s4]

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
        
        # Viewpoints visualizer
        self.viewpoints_pub = rospy.Publisher("/slam/vis_viewpoints",MarkerArray,queue_size=1)

        self.h_lines_pub = rospy.Publisher("/slam/vis_h_lines",MarkerArray,queue_size=1)

        self.tf_br = TransformBroadcaster()
    
    #_______________________   Predictions __________________________________________________________________

    def State_model (self,msg):
        # print(msg.name[0])

        if msg.name[0] == self.wheel_name_left:
            self.left_wheel_velocity = msg.velocity[0]
            # print('got left wheel')
            # self.left_wheel_received = True
                # return
        elif msg.name[0] == self.wheel_name_right:
            self.right_wheel_velocity = msg.velocity[0]
            # print('got right wheel')
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

        # if update is running don't run prediction
        # if not self.update_running:
        dt = self.State_model(msg) 

        Ak = np.array([[1, 0, -self.v * dt * np.sin(self.xk[-1])],
                    [0, 1, self.v * dt * np.cos(self.xk[-1])],
                    [0, 0, 1]]) 

        Wk = np.array([[dt*self.wheel_radius*np.cos(self.xk[-1])/2,   dt*self.wheel_radius*np.cos(self.xk[-1])/2],
                        [dt*self.wheel_radius*np.sin(self.xk[-1])/2,  dt*self.wheel_radius*np.sin(self.xk[-1])/2],
                        [dt*self.wheel_radius/self.wheel_base_distance,    -dt*self.wheel_radius/self.wheel_base_distance]])
            
        # print(self.xk[-1])

        F1k, F2k = self.get_F1k_F2k(Ak, Wk)

        # self.Pk = F1k @ self.Pk @ F1k.T  + F2k @ self.Qk @ F2k.T

        # print(self.Pk)

        self.publish_odom_predict(msg)

    #______________________    Update  ________________________________________________________________
    
    def get_h(self, prev_pose, new_pose):
        return compounding(pose_inversion(prev_pose), new_pose)

    def get_scan(self, scan_msg):
        
        # self.update_running = True

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

        print('curr scan: ', curr_scan)
        # print('scan len: ', curr_scan.shape[0])
        # print('xk: ', self.xk)
        # print('curr_scan: ', curr_scan[-5:])

        # if scan available, then proceed.
        # if curr_scan != []:
        if len(self.xk) == 3 : #add initial scan
            # add new scan
            # print('adding first scan')
            self.scans.append(curr_scan)
            self.xk, self.Pk = AddNewPose(self.xk, self.Pk)
            # self.add_new_pose()
        else:
            last_scan_pose = self.xk[-6:-3]  # 2nd last state in state vector
            curr_pose = self.xk[-3:]         # last state in state vector
            
            dist_since_last_scan = euclidean_distance(last_scan_pose[:2], curr_pose[:2]) 
            rot_since_last_scan = abs(last_scan_pose[2] - curr_pose[2])

            # print('dist_since_last_scan: ', dist_since_last_scan)
            # print('rot_since_last_scan: ', rot_since_last_scan)

            # only add pose/scan if we have move significantly
            if dist_since_last_scan > self.dist_th: #or rot_since_last_scan > self.ang_th:
                # add new scan
                self.scans.append(curr_scan)
                self.xk, self.Pk = AddNewPose(self.xk, self.Pk)
                # self.add_new_pose()
        
        # self.update_running = False
        self.publish_viewpoints()
        self.check_obs_model()

    # def add_new_pose(self):
    #     # print('In add new pose')
    #     new_x = np.zeros(len(self.xk)+3)
    #     # print(self.xk.shape)
    #     # print(new_x.shape)
    #     for i in range(len(self.xk)):
    #         new_x[i] = self.xk[i]
    #     new_x[-1] = self.xk[-1]
    #     new_x[-2] = self.xk[-2]
    #     new_x[-3] = self.xk[-3]

    #     self.xk = new_x
        
    #     self.update_running = False

    ##################      Publishing   ##############################

    #########################-_________________________________________________________________________________________________-##########################################
    
    def check_obs_model(self):
        ref_list = []
        h_list = []

        for i in range(0,len(self.xk)-3,3):
            ref_pose = self.xk[i:i+3]
            next_pose = self.xk[i+3:i+6]
            # x, y, theta
            h = self.get_h(ref_pose, next_pose)
            # print('------------------')
            # print('ref pose: ', ref_pose)
            # print('next_pose: ', next_pose)
            # print('h: ', h)

            ref_list.append(ref_pose)
            h_list.append(h)
        
        self.publish_h_lines(ref_list, h_list)

    def publish_h_lines(self, ref_list, h_list):

        viewpoints_list = []

        # Loop through the data and create markers
        for i in range(len(ref_list)-1):
            # Create a marker for each line segment
            myMarker = Marker()
            myMarker.header.frame_id = "world"
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
            myMarker.header.frame_id = "world"
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
                        # self.myMarker.color=ColorRGBA(colors[i*col_jump,0], colors[val*col_jump,1], colors[val*col_jump,2], 0.5)

            myMarker.scale.x = 0.1
            myMarker.scale.y = 0.1
            myMarker.scale.z = 0.05
            # self.myMarker.lifetime = rospy.Duration(0)
            viewpoints_list.append(myMarker)

        self.viewpoints_pub.publish(viewpoints_list)

    def publish_odom_predict(self,msg):
        current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
        q = quaternion_from_euler(0, 0, self.xk[-1])
        
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

    





