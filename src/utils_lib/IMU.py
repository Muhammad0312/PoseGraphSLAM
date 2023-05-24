import math

def calculate_heading(msg):
    # Extract quaternion orientation
    # q_x = imu_msg['orientation']['x']
    # q_y = imu_msg['orientation']['y']
    # q_z = imu_msg['orientation']['z']
    # q_w = imu_msg['orientation']['w']

    orientation = msg.orientation
    orientation_x = orientation.x
    orientation_y = orientation.y
    orientation_z = orientation.z
    orientation_w = orientation.w

    # Convert quaternion to Euler angles
    roll, pitch, yaw = euler_from_quaternion(orientation_x, orientation_y, orientation_z, orientation_w)
    # print("yaw",yaw)
    return yaw


def euler_from_quaternion(x, y, z, w):
    """
    Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw).
    Taken from: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw
