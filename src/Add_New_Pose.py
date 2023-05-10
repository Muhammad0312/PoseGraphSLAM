import numpy as np

def AddNewPose(x, P):
    # print("P",P.shape)
    n = x.shape[0] + 3
    P_new = np.zeros((P.shape[0], P.shape[0]))
    # print("P_new",P_new.shape)
    x_new = np.zeros((n, 1))

    x_new[:n-3, :] = x
    P_new[:P.shape[0], :P.shape[0]] = P
    x_new[n-3:, :] = np.array([[0], [0], [0]])

    last_col = P[:, -3:]
    # print("last_col",last_col)
    P_new = np.hstack((P_new, last_col))
    last_row = P_new[-3:, :]
    # print("last_row",last_row)
    P_new = np.vstack((P_new, last_row))
    # print("P_new",P_new.shape)

    return x_new, P_new

 
x = np.array([[1], [2], [0]])
P = np.array([[1,2,3], [4,5,6], [7,8,9]])

for i in range(3):
    x, P = AddNewPose(x, P)
    print(f"x =\n{x}")
    print(f"P =\n{P}")
    print("------")
