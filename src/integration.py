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
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from sensor_msgs import point_cloud2



class PoseGraphSLAM:
    def __init__(self) -> None:

        # robot constants
        self.wheel_radius = 0.035
        self.wheel_base_distance = 0.230
        self.update_running = False

        # Pose initialization
        self.xk = np.zeros(3)

        # initial covariance matrix
        self.Pk = np.array([[0.0, 0, 0],    
                            [0, 0.0, 0],
                            [0, 0, 0.0]])     
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
        self.dist_th = 0.1   # take scan if displacement is > 0.2m, 0.5
        self.ang_th = 0.785 # take scan if angle change is > 0.175(10 degrees), 0.785 (45 degrees) 

        self.map = []  # = [s1, s2, s3, s4]

        self.last_time = rospy.Time.now()

        # If using turtlebot_hoi<>.launch _________________
        self.js_sub = rospy.Subscriber("/turtlebot/joint_states", JointState, self.predict)
        self.odom_pub = rospy.Subscriber("/turtlebot/kobuki/sensors/rplidar", LaserScan, self.scan_available)
        
        self.child_frame_id = "turtlebot/kobuki/base_footprint"
        self.wheel_name_left = "turtlebot/kobuki/wheel_left_joint"
        self.wheel_name_right = "turtlebot/kobuki/wheel_right_joint"


        # odom publisher
        self.odom_pub = rospy.Publisher("kobuki/odom", Odometry, queue_size=10)
        
        # Viewpoints visualizer
        self.viewpoints_pub = rospy.Publisher("/slam/vis_viewpoints",MarkerArray,queue_size=1)
        # self.h_lines_pub = rospy.Publisher("/slam/vis_h_lines",MarkerArray,queue_size=1)
        # self.guess_displacement_pub = rospy.Publisher("guess_displacement", Odometry, queue_size=10)

        # # Publisher for icp_displacement
        # self.icp_displacement_pub = rospy.Publisher("icp_displacement", Odometry, queue_size=10)



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
        self.w = (left_lin_vel-right_lin_vel) / self.wheel_base_distance
    
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
     #########################-____________________________-##########################################

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
    

    def scan_available(self,scan_msg):

        scan = get_scan(scan_msg) 

        if scan != []:
            if len(self.xk) == 3 : #add initial scan
                self.xk, self.Pk = AddNewPose(self.xk, self.Pk)
                self.map.append(np.array(scan))
            else:

                is_add_scan = check_distance_bw_scans(self.xk, self.dist_th, self.ang_th)

                # only add pose/scan if we have move significantly
                if is_add_scan: 
                    # add new scan and pose
                    self.xk, self.Pk = AddNewPose(self.xk, self.Pk)
                    self.map.append(np.array(scan))

                    # Overlapping Scans
                    Ho = OverlappingScans(self.xk, self.map)
                    print('Num scans: ', len(self.map))
                    print('Overlap Ho: ', Ho)
                    Z_matched=[]
                    h=[]
                    # for each matched pair
                    for j in Ho:
                        print('------------- mathcing started ---------')
                        print('scan index: ', j)
                        match_scan = self.map[j]
                        
                        curr_viewpoint = self.xk[-3:]
                        matched_viewpoint = self.xk[j*3:3*j+3]

                        # Obervation Model
                        guess_displacement = get_h(curr_viewpoint, matched_viewpoint)
                        # P = J bla bla
                        zr, Rr = icp(match_scan, self.map[-1], matched_viewpoint, curr_viewpoint,guess_displacement)
                        print('guess_displacement: ', guess_displacement)
                        print('icp displacement: ', zr)
                        # if Rr[-1] <= 5:
                        #     print('guess_displacement: ', guess_displacement)
                        #     print('icp displacement: ', zr)
                        h.append(guess_displacement)

                        Z_matched.append(zr)
                        #     guess_displacement_msg = Odometry()
                        #     guess_displacement_msg.header.frame_id = "world_ned"
                        #     guess_displacement_msg.pose.pose.position.x = guess_displacement[0]
                        #     guess_displacement_msg.pose.pose.position.y = guess_displacement[1]

                        #     # Set the orientation quaternion based on the theta angle
                        #     theta = guess_displacement[2]
                        #     quaternion = quaternion_from_euler(0, 0, theta)  # Assuming theta represents yaw
                        #     guess_displacement_msg.pose.pose.orientation.x = quaternion[0]
                        #     guess_displacement_msg.pose.pose.orientation.y = quaternion[1]
                        #     guess_displacement_msg.pose.pose.orientation.z = quaternion[2]
                        #     guess_displacement_msg.pose.pose.orientation.w = quaternion[3]

                        #     self.guess_displacement_pub.publish(guess_displacement_msg)

                        #     icp_displacement_msg = Odometry()
                        #     icp_displacement_msg.header.frame_id = "world_ned"
                        #     icp_displacement_msg.pose.pose.position.x = zr[0]
                        #     icp_displacement_msg.pose.pose.position.y = zr[1]

                        #     # Set the orientation quaternion based on the theta angle
                        #     theta = zr[2]
                        #     quaternion = quaternion_from_euler(0, 0, theta)  # Assuming theta represents yaw
                        #     icp_displacement_msg.pose.pose.orientation.x = quaternion[0]
                        #     icp_displacement_msg.pose.pose.orientation.y = quaternion[1]
                        #     icp_displacement_msg.pose.pose.orientation.z = quaternion[2]
                        #     icp_displacement_msg.pose.pose.orientation.w = quaternion[3]

                        #     self.icp_displacement_pub.publish(icp_displacement_msg)
                        # else:
                        #     pass
                    h = sum(h, [])
                    Z_matched = sum(Z_matched, []) # to convert z_matched from [[],[],[]] to []
                    Zk, Rk, Hk, Vk = ObservationMatrix(Ho, self.xk, Z_matched, Rp=None) # hp = ho for now, Rp=None for now 
                    self.xk, self.Pk = Update(self.xk, self.Pk, Zk, Rk, Hk, Vk,h)
                    print("self.xk",self.xk)


        # self.update_running = False
        self.publish_viewpoints()
        self.check_obs_model()


    ##################      Publishing   ##############################

    #########################-_________________________________________________________________________________________________-##########################################
    
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
        
        covar = [self.Pk[0,0], self.Pk[0,1], 0.0, 0.0, 0.0, self.Pk[0,2],
                self.Pk[1,0], self.Pk[1,1], 0.0, 0.0, 0.0, self.Pk[1,2],  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                self.Pk[2,0], self.Pk[2,1], 0.0, 0.0, 0.0, self.Pk[2,2]]

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

    




