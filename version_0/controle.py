import numpy as np
from scipy.linalg import pinv, null_space
from jak import jak
from invKIN import invKIN
from v_0.cables_v0 import cables

#Including the global variables in an args array allows for more flexibility and makes the function more modular. 
#args = [l1, l2, l3, Lb, r, La, t_, E_, TT, lj, XL, J1, J2, J3, m1, m2, m3, g ]

def controle(t, x, args):
    # Global variables
    #global l1, l2, l3, Lb, r, La, t_, E_, TT
    #global lj, XL

    l1, l2, l3, Lb, r, La, t_, E_, TT, lj, XL, J1, J2, J3, m1, m2, m3, g = args
    Dt = t - t_

    # Ensure angles are between 0 and 2*pi
    for i in range(3):
        x[i] = np.arctan2(np.sin(x[i]), np.cos(x[i]))
        if x[i] < 0:
            x[i] += 2 * np.pi

    q = x[:3]
    dq = x[3:6]

    # Get the cables parameters
    X, Jk, dt_Jk, S_ = cables(lj, XL, q, dq)

    # Trajectory planning
    Xd, V, A = planning(t, lj, q, dq, X, Dt)

    # Control gains
    Kp = 750
    Kv = 2 * np.sqrt(Kp)

    # Error calculation
    delt_X = Xd - X
    delt_X_t = V - Jk @ dq
    Xd_tt = A

    # Dynamic parameters
    xMc, vFc = dyn_0(x)

    # Integral error update
    if Dt != 0:
        E_ += delt_X

    ion = 0

    # Desired acceleration
    X_tt = Xd_tt + Kv * delt_X_t + Kp * delt_X + ion * 0.01 * 10**4 * E_ * Dt

    # Control torque
    tau = xMc @ (pinv(Jk) @ (X_tt - dt_Jk @ dq)) + vFc

    # Force calculation
    Taw = Jk.T @ S_
    pinv_Taw = pinv(Taw) @ tau
    null_Taw = null_space(Taw)

    # Force saturation
    Tmax = 500
    T = pinv_Taw
    c = 1

    while any(T < 0) and all(T < Tmax):
        T += c * null_Taw
        if all(null_Taw <= 0):
            c = -1
        else:
            c = 1

    # Output torque
    Tau = r * T
    F = 0

    # Saturation limits
    maxtau = np.inf + 400 * r
    for i in range(4):
        if abs(Tau[i]) > maxtau:
            Tau[i] = np.sign(Tau[i]) * maxtau

    # Update time history
    if t > t_:
        T = np.concatenate(([t], T, X[:2], X[2:4]))
        TT = np.hstack((TT, T))

    t_ = t
    return Tau, F

def dyn_0(x):
    global J1, J2, J3, m1, m2, m3, l1, l2, l3, g
    q1, q2, q3 = x[:3]
    dq1, dq2, dq3 = x[3:6]

    # Inertia matrix
    xMc = np.array([
        [J2 + J3 + m2 * l1 * l2 * np.cos(q2) + 2 * m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + m3 * l1 * l3 * np.cos(q2 + q3) + J1 + (1/4) * m1 * l1**2 + m3 * l2**2 + (1/4) * m3 * l3**2 + m2 * l1**2 + m3 * l1**2 + (1/4) * m2 * l2**2, J2 + J3 + (1/2) * m2 * l1 * l2 * np.cos(q2) + m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + (1/2) * m3 * l1 * l3 * np.cos(q2 + q3) + m3 * l2**2 + (1/4) * m3 * l3**2 + (1/4) * m2 * l2**2, J3 + (1/2) * m3 * l2 * l3 * np.cos(q3) + (1/2) * m3 * l1 * l3 * np.cos(q2 + q3) + (1/4) * m3 * l3**2],
        [J2 + J3 + (1/2) * m2 * l1 * l2 * np.cos(q2) + m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + (1/2) * m3 * l1 * l3 * np.cos(q2 + q3) + m3 * l2**2 + (1/4) * m3 * l3**2 + (1/4) * m2 * l2**2, m3 * l2 * l3 * np.cos(q3) + (1/4) * m2 * l2**2 + m3 * l2**2 + (1/4) * m3 * l3**2 + J2 + J3, (1/2) * m3 * l2 * l3 * np.cos(q3) + (1/4) * m3 * l3**2 + J3],
        [J3 + (1/2) * m3 * l2 * l3 * np.cos(q3) + (1/2) * m3 * l1 * l3 * np.cos(q2 + q3) + (1/4) * m3 * l3**2, (1/2) * m3 * l2 * l3 * np.cos(q3) + (1/4) * m3 * l3**2 + J3, J3 + (1/4) * m3 * l3**2]
    ])

    # Coriolis and gravity forces
    vFc = np.array([
        -(1/2) * m2 * l1 * l2 * dq2**2 * np.sin(q2) - m3 * l1 * l2 * dq2**2 * np.sin(q2) - (1/2) * m3 * l1 * l3 * dq2**2 * np.sin(q2 + q3) - m2 * l1 * dq1 * l2 * np.sin(q2) * dq2 - 2 * m3 * l1 * dq1 * l2 * np.sin(q2) * dq2 - m3 * l1 * dq1 * l3 * np.sin(q2 + q3) * dq2 - m3 * l1 * l3 * dq3 * np.sin(q2 + q3) * dq2 - m3 * l2 * dq2 * l3 * np.sin(q3) * dq3 - m3 * l1 * dq1 * l3 * np.sin(q2 + q3) * dq3 - m3 * l2 * dq1 * l3 * np.sin(q3) * dq3 - (1/2) * m3 * l1 * l3 * dq3**2 * np.sin(q2 + q3) - (1/2) * m3 * l2 * l3 * dq3**2 * np.sin(q3) + (1/2) * m3 * g * l3 * np.cos(q1 + q2 + q3) + (1/2) * m1 * g * l1 * np.cos(q1) + m2 * g * l1 * np.cos(q1) + m3 * g * l1 * np.cos(q1) + (1/2) * m2 * g * l2 * np.cos(q1 + q2) + m3 * g * l2 * np.cos(q1 + q2),
        -m3 * l2 * dq2 * l3 * np.sin(q3) * dq3 + (1/2) * np.sin(q2) * dq1**2 * l1 * l2 * m2 + np.sin(q2) * dq1**2 * l1 * l2 * m3 + (1/2) * dq1**2 * np.sin(q2 + q3) * l1 * l3 * m3 - m3 * l2 * dq1 * l3 * np.sin(q3) * dq3 - (1/2) * m3 * l2 * l3 * dq3**2 * np.sin(q3) + (1/2) * m3 * g * l3 * np.cos(q1 + q2 + q3) + (1/2) * m2 * g * l2 * np.cos(q1 + q2) + m3 * g * l2 * np.cos(q1 + q2),
        (1/2) * m3 * l3 * (dq2**2 * np.sin(q3) * l2 + 2 * dq2 * l2 * dq3 * np.sin(q3) + dq1**2 * np.sin(q2 + q3) * l1 + dq2**2 * np.sin(q2 + q3) * l1 + 2 * dq1 * l1 * dq3 * np.sin(q2 + q3) + dq1 * l1 * dq2 * np.sin(q2 + q3)) + (1/2) * m3 * g * l3 * np.cos(q1 + q2 + q3)
    ])

    return xMc, vFc
