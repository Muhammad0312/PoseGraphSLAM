import numpy as np

def AddNewPose(x, P):
    print('add new pose ')
    x = np.array(x).reshape(len(x),1)
    n = x.shape[0] + 3
    P_new = np.zeros((P.shape[0], P.shape[0]))
    x_new = np.zeros((n, 1))

    x_new[:n-3, :] = x
    P_new[:P.shape[0], :P.shape[0]] = P
    # x_new[n-3:, :] = np.array([x[0], [0], [0]])
    x_new[n-3:, :] = np.array([x[-3], x[-2], x[-1]])

    last_col = P[:, -3:]
    # print("last_col",last_col)
    P_new = np.hstack((P_new, last_col))
    last_row = P_new[-3:, :]
    # print("last_row",last_row)
    P_new = np.vstack((P_new, last_row))
    # print("P_new",P_new.shape)

    return x_new.T[0], P_new

 
# # x = np.array([[1], [2], [3]])
# x = np.array([1,2,3,4,5,6])
# P = np.array([[1,2,3,4,5,6], 
#               [7,8,9,10,11,12], 
#               [13,14,15,16,17,18], 
#               [19,20,21,22,23,24],
#               [25,26,27,28,29,30],
#               [31,32,33,34,35,36]])

# for i in range(1):
#     x, P = AddNewPose(x, P)
#     print(f"x =\n{x}")
#     print(f"P =\n{P}")
#     print("------")
