import numpy as np
from Mass_matrix import mass_matrix

def RRode(t, x, g, m1, m2, m3, l1, l2, l3, La, Lb, r, TT, t_, lj, XL):
    xdot = np.zeros(6)
    q = np.array([x[0], x[1], x[2]])
    dq = np.array([x[3], x[4], x[5]])

    # Assume cables and controle functions are defined elsewhere and imported
    _, Jk, _, S_ = cables(lj, XL, q, dq)
    Tau, _ = controle(t, x)
    T = (1 / r) * Tau

    vFc = np.zeros(3)
    vFc[0] = (-(1/2) * m2 * l1 * l2 * dq[1]**2 * np.sin(q[1]) - m3 * l1 * l2 * dq[1]**2 * np.sin(q[1])
              - (1/2) * m3 * l1 * l3 * dq[1]**2 * np.sin(q[1] + q[2]) - m2 * l1 * dq[0] * l2 * np.sin(q[1]) * dq[1]
              - 2 * m3 * l1 * dq[0] * l2 * np.sin(q[1]) * dq[1] - m3 * l1 * dq[0] * l3 * np.sin(q[1] + q[2]) * dq[1]
              - m3 * l1 * l3 * dq[2] * np.sin(q[1] + q[2]) * dq[1] - m3 * l2 * dq[1] * l3 * np.sin(q[2]) * dq[2]
              - m3 * l1 * dq[0] * l3 * np.sin(q[1] + q[2]) * dq[2] - m3 * l2 * dq[0] * l3 * np.sin(q[2]) * dq[2]
              - (1/2) * m3 * l1 * l3 * dq[2]**2 * np.sin(q[1] + q[2]) - (1/2) * m3 * l2 * l3 * dq[2]**2 * np.sin(q[2])
              + (1/2) * m3 * g * l3 * np.cos(q[0] + q[1] + q[2]) + (1/2) * m1 * g * l1 * np.cos(q[0])
              + m2 * g * l1 * np.cos(q[0]) + m3 * g * l1 * np.cos(q[0]) + (1/2) * m2 * g * l2 * np.cos(q[0] + q[1])
              + m3 * g * l2 * np.cos(q[0] + q[1]))

    vFc[1] = (-m3 * l2 * dq[1] * l3 * np.sin(q[2]) * dq[2] + (1/2) * np.sin(q[1]) * dq[0]**2 * l1 * l2 * m2
              + np.sin(q[1]) * dq[0]**2 * l1 * l2 * m3 + (1/2) * dq[0]**2 * np.sin(q[1] + q[2]) * l1 * l3 * m3
              - m3 * l2 * dq[0] * l3 * np.sin(q[2]) * dq[2] - (1/2) * m3 * l2 * l3 * dq[2]**2 * np.sin(q[2])
              + (1/2) * m3 * g * l3 * np.cos(q[0] + q[1] + q[2]) + (1/2) * m2 * g * l2 * np.cos(q[0] + q[1])
              + m3 * g * l2 * np.cos(q[0] + q[1]))

    vFc[2] = ((1/2) * m3 * l3 * (dq[1]**2 * np.sin(q[2]) * l2 + 2 * dq[1] * dq[0] * np.sin(q[2]) * l2
              + dq[0]**2 * np.sin(q[1] + q[2]) * l1 + dq[0]**2 * np.sin(q[2]) * l2 + g * np.cos(q[0] + q[1] + q[2])))

    ivp = np.array([q[0], q[1], q[2], dq[0], dq[1], dq[2]])
    M = mass_matrix(ivp, J1, J2, J3, m1, m2, m3, l1, l2, l3)
    vFc = np.linalg.inv(M).dot(np.concatenate((dq, vFc)))

    vFc = -vFc + Jk.T.dot(S_).dot(T) - 0 * dq

    xdot[0] = x[3]
    xdot[1] = x[4]
    xdot[2] = x[5]
    xdot[3] = vFc[0]
    xdot[4] = vFc[1]
    xdot[5] = vFc[2]

    return xdot

# Note: You will need to define the functions `cables`, `controle`, and `mass_matrix` as well as the global variables `J1`, `J2`, `J3` used in `mass_matrix`.
