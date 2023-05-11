#!/usr/bin/python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float64MultiArray
import sys, select, os
import numpy as np

if os.name == 'nt':
  import msvcrt, time
else:
  import tty, termios

if os.name != 'nt':
    settings = termios.tcgetattr(sys.stdin)

msg = """
Control Your TurtleBot3!
---------------------------
ith joint can be controlled by following
first key is +ve, second is negative:

        e r t y
        d f g h

DON'T press key multiple times
press s to stop!

CTRL-C to quit
"""

e = """
Communications Failed
"""

def getKey():
    if os.name == 'nt':
        timeout = 0.1
        startTime = time.time()
        while(1):
            if msvcrt.kbhit():
                if sys.version_info[0] >= 3:
                    return msvcrt.getch().decode()
                else:
                    return msvcrt.getch()
            elif time.time() - startTime > timeout:
                return ''

    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

# read keyboard keys and publish v and w
class TeleopKey:
    def __init__(self):

        vel_pub = rospy.Publisher('/turtlebot/swiftpro/joint_velocity_controller/command', Float64MultiArray)
        
        joint_vels = [0.0,0.0,0.0,0.0]
        
        
        try:
            print(msg)
            while not rospy.is_shutdown():
                key = getKey()

                # Joint 1 positive
                if key == 'e' :
                    joint_vels[0] += 0.1
                    # lin_vel = np.clip(lin_vel,-max_lin_vel,max_lin_vel)
                    F = Float64MultiArray()
                    F.data = joint_vels
                    vel_pub.publish(F)
                # Joint 1 negative
                elif key == 'd' :
                    joint_vels[0] -= 0.1
                    # lin_vel = np.clip(lin_vel,-max_lin_vel,max_lin_vel)
                    F = Float64MultiArray()
                    F.data = joint_vels
                    vel_pub.publish(F)

                # Joint 2 positive
                elif key == 'r' :
                    joint_vels[1] += 0.1
                    # lin_vel = np.clip(lin_vel,-max_lin_vel,max_lin_vel)
                    F = Float64MultiArray()
                    F.data = joint_vels
                    vel_pub.publish(F)
                # Joint 2 negative
                elif key == 'f' :
                    joint_vels[1] -= 0.1
                    # lin_vel = np.clip(lin_vel,-max_lin_vel,max_lin_vel)
                    F = Float64MultiArray()
                    F.data = joint_vels
                    vel_pub.publish(F)

                # Joint 3 positive
                elif key == 't' :
                    joint_vels[2] += 0.1
                    # lin_vel = np.clip(lin_vel,-max_lin_vel,max_lin_vel)
                    F = Float64MultiArray()
                    F.data = joint_vels
                    vel_pub.publish(F)
                # Joint 3 negative
                elif key == 'g' :
                    joint_vels[2] -= 0.1
                    # lin_vel = np.clip(lin_vel,-max_lin_vel,max_lin_vel)
                    F = Float64MultiArray()
                    F.data = joint_vels
                    vel_pub.publish(F)

                # Joint 2 positive
                elif key == 'y' :
                    joint_vels[3] += 0.1
                    # lin_vel = np.clip(lin_vel,-max_lin_vel,max_lin_vel)
                    F = Float64MultiArray()
                    F.data = joint_vels
                    vel_pub.publish(F)
                # Joint 2 negative
                elif key == 'h' :
                    joint_vels[3] -= 0.1
                    # lin_vel = np.clip(lin_vel,-max_lin_vel,max_lin_vel)
                    F = Float64MultiArray()
                    F.data = joint_vels
                    vel_pub.publish(F)

                elif key == ' ' or key == 's' :
                    joint_vels = [0.0,0.0,0.0,0.0]
                    F = Float64MultiArray()
                    F.data = joint_vels
                    vel_pub.publish(F)

                else:
                    if (key == '\x03'):
                        break
                print('------------------------')
                print('Joint Velocities',joint_vels)
                
                
        except:
            print(e)

        # self.joint1_vel_pub = rospy.Publisher('joint1_velocity_topic', Float64, queue_size=10)

   
if __name__ == '__main__':
    rospy.init_node('teleop_arm_key')
    velocity_converter = TeleopKey()
    rospy.spin()