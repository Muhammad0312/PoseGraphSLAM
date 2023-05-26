def jacobian_hx_robot(self, scan_index, xk_slam):
        
        #extract the last 3 rows of the xk_slam array and store them in a variable called xk
        xk_robot = xk_slam[-3:, :]

        #extract the x,y and theta of the scan pose from the xk_slam array
        xk_scan = xk_slam[3*scan_index:3*scan_index+3, :]


        robot_pose_jacobian = np.array([[-np.cos(xk_robot[2,0]), -np.sin(xk_robot[2,0]), -np.sin(xk_robot[2,0])(xk_scan[0,0]-xk_robot[0,0]) + np.cos(xk_robot[2,0])(xk_scan[1,0]-xk_robot[1,0])],
                                [np.sin(xk_robot[2,0]), -np.cos(xk_robot[2,0]), -np.cos(xk_robot[2,0])(xk_scan[0,0]-xk_robot[0,0])-np.sin(xk_robot[2,0])(xk_scan[1,0]-xk_robot[1,0])],
                                [0, 0, -1]])
        
        return robot_pose_jacobian

def jacobian_hx_scan(self, xk_slam):

        xk_robot = xk_slam[-3:, :]

        scan_pose_jacobian = np.array([[np.cos(xk_robot[2,0]), np.sin(xk_robot[2,0]), 0],
                             [-np.sin(xk_robot[2,0]), np.cos(xk_robot[2,0]), 0],
                             [0, 0, 1]])
        # scan_pose_jacobian = np.array([[np.cos(xk_robot[2,0]), -np.sin(xk_robot[2,0]), 0],
        #                      [np.sin(xk_robot[2,0]), np.cos(xk_robot[2,0]), 0],
        #                      [0, 0, 1]])
        
        return scan_pose_jacobian

def ICP_jacobianHk_V2(self,scan_index, xk_slam):
        num_poses = int(xk_slam.shape[0]/3)

        #extract the last 3 rows of the xk_slam array and store them in a variable called xk
        xk_robot = xk_slam[-3:, :]

        #extract the x,y and theta of the scan pose from the xk_slam array
        xk_scan = xk_slam[3*scan_index:3*scan_index+3, :]

        #create a zero matrix of size 3 x 3*num_poses
        jacobianHk = np.zeros((3, 3*num_poses))

        #replace the last 3 rows of the last 3 columns of the zero matrix with the dh_dxk
        # jacobianHk[:, -6:-3] = self.jacobian_hx_robot(scan_index, xk_slam)

        jacobianHk[:, -3:] = self.jacobian_hx_robot(scan_index, xk_slam)

        #replace the all the rows of the column from 3*scan_index to 3*scan_index+3 with the j2_plus
        jacobianHk[:, 3*scan_index:3*scan_index+3] = self.jacobian_hx_scan(xk_slam)

        return jacobianHk