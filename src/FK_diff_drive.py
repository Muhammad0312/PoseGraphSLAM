#!/usr/bin/python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float64MultiArray

class VelocityConverter:
    def __init__(self):

        # robot constants
        self.r = 0.035
        self.L = 0.230

        rospy.Subscriber('/lin_ang_velocities', Float64MultiArray, self.velocity_callback)

        self.vel_pub = rospy.Publisher('turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray,queue_size=1)

    def velocity_callback(self, msg):
        data = msg.data
        v = data[0]
        w = data[1]
        
        left_vel = (2*v - w*self.L)/(2*self.r)
        right_vel = (2*v + w*self.L)/(2*self.r)

        F = Float64MultiArray()
        F.data = [left_vel,right_vel]
        self.vel_pub.publish(F)

if __name__ == '__main__':
    rospy.init_node('velocity_converter')
    velocity_converter = VelocityConverter()
    rospy.spin()