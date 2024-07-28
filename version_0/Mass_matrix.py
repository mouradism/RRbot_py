import numpy as np

def Mass_matrix(ivp):
    x, J1, J2, J3, m1, m2, m3, l1, l2, l3=ivp
    q1, q2, q3 ,dq1, dq2, dq3 = x
    
    xMc = np.array([
        [J2 + J3 + m2 * l1 * l2 * np.cos(q2) + 2 * m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + m3 * l1 * l3 * np.cos(q2 + q3) + J1 + 0.25 * m1 * l1**2 + m3 * l2**2 + 0.25 * m3 * l3**2 + m2 * l1**2 + m3 * l1**2 + 0.25 * m2 * l2**2, J2 + J3 + 0.5 * m2 * l1 * l2 * np.cos(q2) + m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + 0.5 * m3 * l1 * l3 * np.cos(q2 + q3) + m3 * l2**2 + 0.25 * m3 * l3**2 + 0.25 * m2 * l2**2, J3 + 0.5 * m3 * l2 * l3 * np.cos(q3) + 0.5 * m3 * l1 * l3 * np.cos(q2 + q3) + 0.25 * m3 * l3**2],
        [J2 + J3 + 0.5 * m2 * l1 * l2 * np.cos(q2) + m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + 0.5 * m3 * l1 * l3 * np.cos(q2 + q3) + m3 * l2**2 + 0.25 * m3 * l3**2 + 0.25 * m2 * l2**2, m3 * l2 * l3 * np.cos(q3) + 0.25 * m2 * l2**2 + m3 * l2**2 + 0.25 * m3 * l3**2 + J2 + J3, 0.5 * m3 * l2 * l3 * np.cos(q3) + 0.25 * m3 * l3**2 + J3],
        [J3 + 0.5 * m3 * l2 * l3 * np.cos(q3) + 0.5 * m3 * l1 * l3 * np.cos(q2 + q3) + 0.25 * m3 * l3**2, 0.5 * m3 * l2 * l3 * np.cos(q3) + 0.25 * m3 * l3**2 + J3, J3 + 0.25 * m3 * l3**2]
    ])

    xMs = np.block([
        [np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 3)), xMc]
    ])

    return xMs