#!/usr/bin/python3

import numpy as np
import rospy, math
from nav_msgs.msg import Odometry 
from std_msgs.msg import Float64, Float32MultiArray
from sensor_msgs.msg import JointState
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

from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from sensor_msgs import point_cloud2
import threading


class PoseGraphSLAM:
    def __init__(self) -> None:

        self.mutex = threading.Lock()
        # robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230
        self.update_running = False

        # Store the map
        self.map = []  # = [s1, s2, s3, s4]
        # Store scan as soon as it is available
        self.scan = []

        # Store groundtruth pose
        self.gt_pose = np.zeros(3)
        # Subscriber to groundtruth
        self.gt_sub = rospy.Subscriber("/turtlebot/stonefish_simulator/ground_truth_odometry", Odometry, self.get_gt)

        # Pose initialization
        self.xk = np.array([0.0, 0.0, 0.0])
        # Groundtruth state vector
        self.gt_xk = np.array([0.0, 0.0, 0.0])

        # initial covariance matrix
        self.Pk = np.array([[0.0, 0, 0],    
                            [0, 0.0, 0],
                            [0, 0, 0.0]])   
        
        # Subscriber to lidar
        self.scan_sub = rospy.Subscriber("/turtlebot/kobuki/sensors/rplidar", LaserScan, self.scan_available)

        # Add new pose to keep predicting
        self.xk, self.Pk = AddNewPose(self.xk, self.Pk)

        self.tf_br = TransformBroadcaster()
        # Subscriber to get joint states
        self.js_sub = rospy.Subscriber("/turtlebot/joint_states", JointState, self.predict)
        # Odometry noise covariance
        self.Qk = np.array([[0.005, 0],     
                             [0, 0.005]])


        # prediction related variables
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.v=0.0
        self.w=0.0
        self.left_wheel_velocity = 0.0
        self.right_wheel_velocity = 0.0
        self.left_wheel_received = False

        # scan related variables
        self.dist_th = 0.03   # take scan if displacement is > 0.2m, 0.5
        self.ang_th = 0.055 # take scan if angle change is > 0.175(10 degrees), 0.785 (45 degrees)

        self.last_time = rospy.Time.now()

        # If using turtlebot_hoi<>.launch _________________
        self.child_frame_id = "turtlebot/kobuki/base_footprint"
        self.wheel_name_left = "turtlebot/kobuki/wheel_left_joint"
        self.wheel_name_right = "turtlebot/kobuki/wheel_right_joint"

        # odom publisher
        self.odom_pub = rospy.Publisher("kobuki/odom", Odometry, queue_size=10)
        
        # Viewpoints visualizer
        self.viewpoints_pub = rospy.Publisher("/slam/vis_viewpoints",MarkerArray,queue_size=1)
        self.full_map_pub = rospy.Publisher('/slam/map', PointCloud2, queue_size=10)

        
    
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
                self.w = (left_lin_vel-right_lin_vel) / self.wheel_base_distance
            
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
                # print('========================')
                # print('F1k: ', F1k.shape)
                # print('Pk: ', self.Pk.shape)
                # print('F1k.T: ', F1k.T.shape)
                # print('F2k: ', F2k.shape)
                # print('Qk: ', self.Qk.shape)
                # print('F2k.T: ', F2k.T.shape)
                self.Pk = F1k @ self.Pk @ F1k.T  + F2k @ self.Qk @ F2k.T
                # print('Pk Updated')

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
        with self.mutex:
            self.State_model(msg) 
            self.publish_odom_predict(msg)

    #______________________    Update  ________________________________________________________________
    

    def scan_available(self,scan_msg):
        self.scan = get_scan(scan_msg)
        if len(self.map) == 0:
            self.map.append(self.scan)
        
        if check_distance_bw_scans(self.xk, self.dist_th, self.ang_th):
            with self.mutex:
                # add new scan and pose
                self.xk, self.Pk = AddNewPose(self.xk, self.Pk)
                self.map.append(np.array(self.scan))

                # Store the actual viewpoint in the groundtruth state vector
                self.gt_xk = np.hstack((self.gt_xk, self.gt_pose))

                # print('Ground truth state vector: ', self.gt_xk)

                # print('Xk inside scan: ',self.xk.shape)
                # print('Pk inside scan: ',self.Pk.shape)
                # print('Number of scans inside scan: ',len(self.map))
                # Overlapping Scans
                Ho = OverlappingScans(self.xk[0:-3], self.map)
                # print('Num scans inside scan: ', len(self.map))
                print('Overlap Ho inside scan: ', Ho)
                Z_matched=[]
                h=[]
                # for each matched pair
                for j in Ho:
                    # print('------------- mathcing started ---------')
                    # print('scan index: ', j)
                    match_scan = self.map[j]
                    
                    curr_viewpoint = self.xk[-6:-3]
                    matched_viewpoint = self.xk[j*3: 3*j+3]

                    curr_viewpoint_gt = self.gt_xk[-3: ]
                    matched_viewpoint_gt = self.gt_xk[j*3: 3*j+3]

                    # Obervation Model
                    guess_displacement = get_h(curr_viewpoint, matched_viewpoint)
                    actual_displacement = get_h(curr_viewpoint_gt, matched_viewpoint_gt)
                    # P = J bla bla
                    # zr, Rr = icp(match_scan, self.map[-1], matched_viewpoint, curr_viewpoint,guess_displacement)
                    zr, Rr = icp(match_scan, self.map[-1], matched_viewpoint, curr_viewpoint)
                    
                    # Suppress scientific notations while displaying numbers
                    np.set_printoptions(suppress=True)
                    print('===================================================')
                    print('Actual displacement: ', np.round(actual_displacement, 6))
                    print('Expected result: ', np.round(guess_displacement, 6))
                    print('ICP: ', np.round(zr, 6))
                    
                    h.append(guess_displacement)
                    Z_matched.append(zr) 
                h = sum(h, [])
                Z_matched = sum(Z_matched, []) # to convert z_matched from [[],[],[]] to []
                Zk, Rk, Hk, Vk = ObservationMatrix(Ho, self.xk, Z_matched, Rp=None) # hp = ho for now, Rp=None for now 
                self.xk, self.Pk = Update(self.xk, self.Pk, Zk, Rk, Hk, Vk, h)
        
                # self.mutex.release()
                # self.update_running = False
                # self.publish_viewpoints()
                # self.check_obs_model()
                self.publish_full_map()


    ##################      Publishing   ##############################

    #########################-_________________________________________________________________________________________________-##########################################

    def get_gt(self, msg):
        euler = euler_from_quaternion([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        self.gt_pose[0] = msg.pose.pose.position.x
        self.gt_pose[1] = msg.pose.pose.position.y
        self.gt_pose[2] = euler[-1]



    def publish_full_map(self):
        # print('State vector: ',self.xk.shape)
        # print('Map: ', len(self.map))
        full_map = scans_to_map(self.xk, self.map)
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
            # print('------------------')
            # print('ref pose: ', ref_pose)
            # print('next_pose: ', next_pose)
            # print('h: ', h)

            ref_list.append(ref_pose)
            h_list.append(h)
        
        # self.publish_h_lines(ref_list, h_list)

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

    #______________________________________________________________________-##########################################

if __name__ == '__main__':

    rospy.init_node("PoseGraphSLAM")

    robot = PoseGraphSLAM()

    rospy.spin()

    




