import numpy as np


def Mass_matrix(x,Args):
    # Extract variables
    J1 = Args["J1"]
    J2 = Args["J2"]
    J3 = Args["J3"]
    m1 = Args["m1"]
    m2 = Args["m2"]
    m3 = Args["m3"]
    l1 = Args["l1"]
    l2 = Args["l2"]
    l3 = Args["l3"]

    # Extract joint angles and their derivatives from input vector x
    x=x.reshape(6)
    q1, q2, q3 = x[0], x[1], x[2]
    dq1, dq2, dq3 = x[3], x[4], x[5]

    # Compute elements of the mass matrix
    xMc = np.array([
        [J2 + J3 + m2 * l1 * l2 * np.cos(q2) + 2 * m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + m3 * l1 * l3 * np.cos(q2 + q3) + J1 + 0.25 * m1 * l1**2 + m3 * l2**2 + 0.25 * m3 * l3**2 + m2 * l1**2 + m3 * l1**2 + 0.25 * m2 * l2**2,
         J2 + J3 + 0.5 * m2 * l1 * l2 * np.cos(q2) + m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + 0.5 * m3 * l1 * l3 * np.cos(q2 + q3) + m3 * l2**2 + 0.25 * m3 * l3**2 + 0.25 * m2 * l2**2,
         J3 + 0.5 * m3 * l2 * l3 * np.cos(q3) + 0.5 * m3 * l1 * l3 * np.cos(q2 + q3) + 0.25 * m3 * l3**2],
        [J2 + J3 + 0.5 * m2 * l1 * l2 * np.cos(q2) + m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + 0.5 * m3 * l1 * l3 * np.cos(q2 + q3) + m3 * l2**2 + 0.25 * m3 * l3**2 + 0.25 * m2 * l2**2,
         m3 * l2 * l3 * np.cos(q3) + 0.25 * m2 * l2**2 + m3 * l2**2 + 0.25 * m3 * l3**2 + J2 + J3,
         0.5 * m3 * l2 * l3 * np.cos(q3) + 0.25 * m3 * l3**2 + J3],
        [J3 + 0.5 * m3 * l2 * l3 * np.cos(q3) + 0.5 * m3 * l1 * l3 * np.cos(q2 + q3) + 0.25 * m3 * l3**2,
         0.5 * m3 * l2 * l3 * np.cos(q3) + 0.25 * m3 * l3**2 + J3,
         J3 + 0.25 * m3 * l3**2]
    ])

    # Assemble the mass matrix
    xMs = np.block([
        [np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), xMc]
    ])

    return xMs , xMc
