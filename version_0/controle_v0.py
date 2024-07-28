import numpy as np
from scipy.linalg import pinv, null_space
from jak import jak
from invKIN import invKIN
from v_0.cables_v0 import cables


# Define global variables
g = m1 = m2 = m3 = l1 = l2 = l3 = La = Lb = r = TT = t_ = lj = XL = E_ = TT = None

def controle(t, x):
    global l1, l2, l3, Lb, r, La, t_, E_, TT, lj, XL
    Dt = t - t_

    x[0] = np.arctan2(np.sin(x[0]), np.cos(x[0]))
    x[1] = np.arctan2(np.sin(x[1]), np.cos(x[1]))
    x[2] = np.arctan2(np.sin(x[2]), np.cos(x[2]))

    if x[0] < 0:
        x[0] += 2 * np.pi
    if x[1] < 0:
        x[1] += 2 * np.pi
    if x[2] < 0:
        x[2] += 2 * np.pi

    q = np.array([x[0], x[1], x[2]])
    dq = np.array([x[3], x[4], x[5]])

    X, Jk, dt_Jk, S_ = cables(lj, XL, q, dq,l1, l2, l3)
    Xd, V, A = planning(t, lj, q, dq, X, Dt)

    Kp = 750
    Kv = 2 * np.sqrt(Kp)

    delt_X = Xd - X
    delt_X_t = V - Jk @ dq
    Xd_tt = A

    xMc, vFc = dyn_0(x)

    if Dt != 0:
        E_ += delt_X

    ion = 0

    X_tt = Xd_tt + Kv * delt_X_t + Kp * delt_X + ion * 0.01 * 10**4 * E_ * Dt
    tau = xMc @ ((pinv(Jk) @ (X_tt - dt_Jk @ dq))) + vFc
    Taw = Jk.T @ S_

    pinv_Taw = pinv(Taw) @ tau
    null_Taw = null_space(Taw)

    Tmax = 500
    T = pinv_Taw
    c = -1 if all(null_Taw <= 0) else 1

    while any(T < 0) and all(T < Tmax):
        T += c * null_Taw

    Tau = r * T
    F = 0

    maxtau = float('inf') + 400 * r

    Tau = np.clip(Tau, -maxtau, maxtau)

    if t > t_:
        T_log = np.vstack((t, T, X[:2], X[2:4]))
        TT = np.hstack((TT, T_log))

    t_ = t
    return Tau, F

def dyn_0(x):
    global J1, J2, J3, m1, m2, m3, l1, l2, l3, g
    q1, q2, q3, dq1, dq2, dq3 = x

    xMc = np.array([
        [J2 + J3 + m2 * l1 * l2 * np.cos(q2) + 2 * m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + m3 * l1 * l3 * np.cos(q2 + q3) + J1 + (1/4) * m1 * l1**2 + m3 * l2**2 + (1/4) * m3 * l3**2 + m2 * l1**2 + m3 * l1**2 + (1/4) * m2 * l2**2, J2 + J3 + (1/2) * m2 * l1 * l2 * np.cos(q2) + m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + (1/2) * m3 * l1 * l3 * np.cos(q2 + q3) + m3 * l2**2 + (1/4) * m3 * l3**2 + (1/4) * m2 * l2**2, J3 + (1/2) * m3 * l2 * l3 * np.cos(q3) + (1/2) * m3 * l1 * l3 * np.cos(q2 + q3) + (1/4) * m3 * l3**2],
        [J2 + J3 + (1/2) * m2 * l1 * l2 * np.cos(q2) + m3 * l1 * l2 * np.cos(q2) + m3 * l2 * l3 * np.cos(q3) + (1/2) * m3 * l1 * l3 * np.cos(q2 + q3) + m3 * l2**2 + (1/4) * m3 * l3**2 + (1/4) * m2 * l2**2, m3 * l2 * l3 * np.cos(q3) + (1/4) * m2 * l2**2 + m3 * l2**2 + (1/4) * m3 * l3**2 + J2 + J3, (1/2) * m3 * l2 * l3 * np.cos(q3) + (1/4) * m3 * l3**2 + J3],
        [J3 + (1/2) * m3 * l2 * l3 * np.cos(q3) + (1/2) * m3 * l1 * l3 * np.cos(q2 + q3) + (1/4) * m3 * l3**2, (1/2) * m3 * l2 * l3 * np.cos(q3) + (1/4) * m3 * l3**2 + J3, J3 + (1/4) * m3 * l3**2]
    ])
    
    vFc = np.array([
        -(1/2) * m2 * l1 * l2 * dq2**2 * np.sin(q2) - m3 * l1 * l2 * dq2**2 * np.sin(q2) - (1/2) * m3 * l1 * l3 * dq2**2 * np.sin(q2 + q3) - m2 * l1 * dq1 * l2 * np.sin(q2) * dq2 - 2 * m3 * l1 * dq1 * l2 * np.sin(q2) * dq2 - m3 * l1 * dq1 * l3 * np.sin(q2 + q3) * dq2 - m3 * l1 * l3 * dq3 * np.sin(q2 + q3) * dq2 - m3 * l2 * dq2 * l3 * np.sin(q3) * dq3 - m3 * l1 * dq1 * l3 * np.sin(q2 + q3) * dq3 - m3 * l2 * dq1 * l3 * np.sin(q3) * dq3 - (1/2) * m3 * l1 * l3 * dq3**2 * np.sin(q2 + q3) - (1/2) * m3 * l2 * l3 * dq3**2 * np.sin(q3) + (1/2) * m3 * g * l3 * np.cos(q1 + q2 + q3) + (1/2) * m1 * g * l1 * np.cos(q1) + m2 * g * l1 * np.cos(q1) + m3 * g * l1 * np.cos(q1) + (1/2) * m2 * g * l2 * np.cos(q1 + q2) + m3 * g * l2 * np.cos(q1 + q2),
        -m3 * l2 * dq2 * l3 * np.sin(q3) * dq3 + (1/2) * np.sin(q2) * dq1**2 * l1 * l2 * m2 + np.sin(q2) * dq1**2 * l1 * l2 * m3 + (1/2) * dq1**2 * np.sin(q2 + q3) * l1 * l3 * m3 - m3 * l2 * dq1 * l3 * np.sin(q3) * dq3 - (1/2) * m3 * l2 * l3 * dq3**2 * np.sin(q3) + (1/2) * m3 * g * l3 * np.cos(q1 + q2 + q3) + (1/2) * m2 * g * l2 * np.cos(q1 + q2) + m3 * g * l2 * np.cos(q1 + q2),
        (1/2) * m3 * l3 * (dq2**2 * np.sin(q3) * l2 + 2 * dq2 * dq1 * np.sin(q3) * l2 + dq1**2 * np.sin(q2 + q3) * l1 + dq1**2 * np.sin(q3) * l2 + g * np.cos(q1 + q2 + q3))
    ])
    
    vFc = vFc - 0 * np.array([-1 * dq1, -1 * dq2, -0 * dq3])
    
    return xMc, vFc

def planning(t, lj, q, dq, X, Dt):
    global l1, l2, l3, ivp, alpha, Xi, Xf
    
    def section_lengths(lj, l1, l2, l3):
        if lj <= l1:
            return lj, 0, 0
        elif lj <= l1 + l2:
            return l1, lj - l1, 0
        elif lj <= l1 + l2 + l3:
            return l1, l2, lj - (l1 + l2)
        else:
            return None, None, None

    l11, l12, l13 = section_lengths(lj[0], l1, l2, l3)
    l21, l22, l23 = section_lengths(lj[1], l1, l2, l3)
    l31, l32, l33 = section_lengths(lj[2], l1, l2, l3)
    l41, l42, l43 = section_lengths(lj[3], l1, l2, l3)

    Vi = np.array([0, 0])
    Vf = np.array([0, 0])
    tf = 1.0

    a = -(2 / tf**3) * (Xf - Xi) + (1 / tf**2) * (Vf + Vi)
    b = (3 / tf**2) * (Xf - Xi) - (2 / tf) * Vi - (1 / tf) * Vf
    c = Vi
    d = Xi

    if t <= tf:
        Xc = a * t**3 + b * t**2 + c * t + d
        Vc = 3 * a * t**2 + 2 * b * t + c
        Ac = 6 * a * t + 2 * b
    else:
        t = tf
        Xc = a * t**3 + b * t**2 + c * t + d
        Vc = 3 * a * t**2 + 2 * b * t + c
        Ac = 6 * a * t + 2 * b

    X3d = Xc
    X2d = Xc + l3 * np.array([np.cos(alpha), np.sin(alpha)])
    V3 = Vc
    V2 = Vc
    A3 = Ac
    A2 = Ac

    qc = invKIN(Xc + l3 * np.array([np.cos(alpha), np.sin(alpha)]), Xc)

    Jkc3, _, _ = jak(l1, l2, l3, qc, np.array([0, 0, 0]))
    Jkc2, _, _ = jak(l1, l2, 0, qc, np.array([0, 0, 0]))
    dqc = (0.5 * pinv(Jkc3) + 0.5 * pinv(Jkc2)) @ Vc

    Jkc3, dt_Jkc3, _ = jak(l1, l2, l3, qc, dqc)
    Jkc2, dt_Jkc2, _ = jak(l1, l2, 0, qc, dqc)

    ddqc = (0.5 * pinv(Jkc3) + 0.5 * pinv(Jkc2)) @ (Ac - (0.5 * dt_Jkc3 + 0.5 * dt_Jkc2) @ qc)

    Jk1, dt_Jk1, X1 = jak(l11, l12, l13, qc, dqc)
    Jk2, dt_Jk2, X2 = jak(l21, l22, l23, qc, dqc)
    Jk3, dt_Jk3, X3 = jak(l31, l32, l33, qc, dqc)
    Jk4, dt_Jk4, X4 = jak(l41, l42, l43, qc, dqc)

    Jk = np.vstack((Jk1, Jk2, Jk3, Jk4))
    dt_Jk = np.vstack((dt_Jk1, dt_Jk2, dt_Jk3, dt_Jk4))

    Xd = np.vstack((X1, X2, X3, X4))
    Vd = Jk @ dqc
    Ad = Jk @ ddqc + dt_Jk @ dqc

    return Xd, Vd, Ad
