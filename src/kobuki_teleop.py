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
Moving around:
        w
   a    s    d
        x

w/x : increase/decrease linear velocity (Burger : ~ 0.22, Waffle and Waffle Pi : ~ 0.26)
a/d : increase/decrease angular velocity (Burger : ~ 2.84, Waffle and Waffle Pi : ~ 1.82)

space key, s : force stop

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

        vel_pub = rospy.Publisher('/lin_ang_velocities', Float64MultiArray)
        
        lin_vel = 0.0
        ang_vel = 0.0
        max_lin_vel = 0.2
        max_ang_vel = 0.35

        try:
            print(msg)
            while not rospy.is_shutdown():
                key = getKey()

                if key == 'w' :
                    lin_vel += 0.025
                    lin_vel = np.clip(lin_vel,-max_lin_vel,max_lin_vel)
                    print('------------------------')
                    print('lin_vel',round(lin_vel,2))
                    print('ang_vel',round(ang_vel,2))
                    F = Float64MultiArray()
                    F.data = [lin_vel,ang_vel]
                    vel_pub.publish(F)
                    
                elif key == 'x' :
                    lin_vel -= 0.025
                    lin_vel = np.clip(lin_vel,-max_lin_vel,max_lin_vel)
                    print('------------------------')
                    print('lin_vel',round(lin_vel,2))
                    print('ang_vel',round(ang_vel,2))
                    F = Float64MultiArray()
                    F.data = [lin_vel,ang_vel]
                    vel_pub.publish(F)

                elif key == 'a' :
                    ang_vel += 0.05
                    ang_vel = np.clip(ang_vel,-max_ang_vel,max_ang_vel)
                    print('------------------------')
                    print('lin_vel',round(lin_vel,2))
                    print('ang_vel',round(ang_vel,2))
                    F = Float64MultiArray()
                    F.data = [lin_vel,ang_vel]
                    vel_pub.publish(F)

                elif key == 'd' :
                    ang_vel -= 0.05
                    ang_vel = np.clip(ang_vel,-max_ang_vel,max_ang_vel)
                    print('------------------------')
                    print('lin_vel',round(lin_vel,2))
                    print('ang_vel',round(ang_vel,2))
                    F = Float64MultiArray()
                    F.data = [lin_vel,ang_vel]
                    vel_pub.publish(F)

                elif key == ' ' or key == 's' :
                    lin_vel = 0.0
                    ang_vel = 0.0
                    print('------------------------')
                    print('lin_vel',round(lin_vel,2))
                    print('ang_vel',round(ang_vel,2))
                    F = Float64MultiArray()
                    F.data = [lin_vel,ang_vel]
                    vel_pub.publish(F)

                else:
                    if (key == '\x03'):
                        break
                
        except:
            print(e)

        # self.joint1_vel_pub = rospy.Publisher('joint1_velocity_topic', Float64, queue_size=10)

   
if __name__ == '__main__':
    rospy.init_node('teleop_key')
    velocity_converter = TeleopKey()
    rospy.spin()