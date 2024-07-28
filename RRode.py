import numpy as np
#Functions for cables, controle, and Mass_matrix
from cables import cables
from controle_perp import controle
from Mass_matrix import Mass_matrix

def computeForces(q, dq,Args):
    g = Args["g"]
    m1, m2, m3 = Args["m1"], Args["m2"], Args["m3"]
    l1, l2, l3 = Args["l1"], Args["l2"], Args["l3"]

    vFc = np.zeros(3)
    # Force calculations for q1
    vFc[0] = -(1/2) * m2 * l1 * l2 * dq[1]**2 * np.sin(q[1]) - \
             m3 * l1 * l2 * dq[1]**2 * np.sin(q[1]) - \
             (1/2) * m3 * l1 * l3 * dq[1]**2 * np.sin(q[1] + q[2]) - \
             m2 * l1 * dq[0] * l2 * np.sin(q[1]) * dq[1] - \
             2 * m3 * l1 * dq[0] * l2 * np.sin(q[1]) * dq[1] - \
             m3 * l1 * dq[0] * l3 * np.sin(q[1] + q[2]) * dq[1] - \
             m3 * l1 * l3 * dq[2] * np.sin(q[1] + q[2]) * dq[1] - \
             m3 * l2 * dq[1] * l3 * np.sin(q[2]) * dq[2] - \
             m3 * l1 * dq[0] * l3 * np.sin(q[1] + q[2]) * dq[2] - \
             m3 * l2 * dq[0] * l3 * np.sin(q[2]) * dq[2] - \
             (1/2) * m3 * l1 * l3 * dq[2]**2 * np.sin(q[1] + q[2]) - \
             (1/2) * m3 * l2 * l3 * dq[2]**2 * np.sin(q[2]) + \
             (1/2) * m3 * g * l3 * np.cos(q[0] + q[1] + q[2]) + \
             (1/2) * m1 * g * l1 * np.cos(q[0]) + \
             m2 * g * l1 * np.cos(q[0]) + \
             m3 * g * l1 * np.cos(q[0]) + \
             (1/2) * m2 * g * l2 * np.cos(q[0] + q[1]) + \
             m3 * g * l2 * np.cos(q[0] + q[1])

    # Force calculations for q2
    vFc[1] = -m3 * l2 * dq[1] * l3 * np.sin(q[2]) * dq[2] + \
              (1/2) * np.sin(q[1]) * dq[0]**2 * l1 * l2 * m2 + \
              np.sin(q[1]) * dq[0]**2 * l1 * l2 * m3 + \
              (1/2) * dq[0]**2 * np.sin(q[1] + q[2]) * l1 * l3 * m3 - \
              m3 * l2 * dq[0] * l3 * np.sin(q[2]) * dq[2] - \
              (1/2) * m3 * l2 * l3 * dq[2]**2 * np.sin(q[2]) + \
              (1/2) * m3 * g * l3 * np.cos(q[0] + q[1] + q[2]) + \
              (1/2) * m2 * g * l2 * np.cos(q[0] + q[1]) + \
              m3 * g * l2 * np.cos(q[0] + q[1])

    # Force calculations for q3
    vFc[2] = (1/2) * m3 * l3 * (dq[1]**2 * np.sin(q[2]) * l2 + \
               2 * dq[1] * dq[0] * np.sin(q[2]) * l2 + \
               dq[0]**2 * np.sin(q[1] + q[2]) * l1 + \
               dq[0]**2 * np.sin(q[2]) * l2 + \
               g * np.cos(q[0] + q[1] + q[2]))
    return vFc

def RRode(t, x ,Args):
    xdot = np.zeros(6)
    q = x[0:3]
    dq = x[3:6]

    # Compute the Jacobian, mass matrix, and other related parameters
    _, Jk, _, S_ = cables(q, dq,Args)

    # Calculate control torques
    Tau, _ = controle(t, x, Args)
    T = (1 / Args["r"]) * Tau

    # Calculate centrifugal and Coriolis forces
    vFc = computeForces(q, dq ,Args)

    # Initial velocity vector
    ivp = np.concatenate((q, dq)).reshape(2,3)

    MassMat,_= Mass_matrix(ivp ,Args)
    # Compute forces and dynamics
    print('--------------------------vvvvvvvvvvvvvv-----------------------------------')
    print((-np.linalg.inv(MassMat) @ np.concatenate((dq, vFc))))
    print(Jk.T @ S_ @ T)
    print('--------------------------^^^^^^^^^^^^^^-----------------------------------')

    vFc = (-np.linalg.inv(MassMat) @ np.concatenate((dq, vFc)))[:3] + Jk.T @ S_ @ T
    # Assign derivatives
    xdot[0:3] = dq
    xdot[3:6] = vFc[0:3]

    return xdot