import numpy as np
from scipy.linalg import pinv, null_space
from jak import jak
from invKIN import invKIN
from cables import cables
from Mass_matrix import Mass_matrix

def controle(t, x, Args):
    ## Extract variables l1, l2, l3, Lb, r, La, t_, E_, TT, lj, XL = Args
    l1 = Args["l1"]
    l2 = Args["l2"]
    l3 = Args["l3"]
    La = Args["La"]
    Lb = Args["Lb"]
    r  = Args["r"]
    t_ = Args["t_"]
    E_ = Args["E_"]
    TT = Args["TT"]
    lj = Args["lj"]
    XL = Args["XL"]

    Dt = t - Args["t_"]

    # Ensure angles are between 0 and 2*pi
    x = np.array([np.arctan2(np.sin(x_i), np.cos(x_i)) for x_i in x])
    x[x < 0] += 2*np.pi

    q = x[:3]
    dq = x[3:6]

    # Get the cables parameters
    X, Jk, dt_Jk, S_ = cables(q, dq, Args)

    # Trajectory planning
    Xd, V, A = planning(t, lj, q, dq, X, Dt, Args)
   
    # Control gains
    Kp = 750
    Kv = 2 * np.sqrt(Kp)

    # Error calculation
    delt_X = Xd - X

    delt_X_t = V - Jk @ dq
    Xd_tt = A

    # Dynamic parameters
    xMc, vFc = dyn_0(x, Args)

    # Integral error update
    if Dt != 0:
        E_ += delt_X

    ion = 0

    # Desired acceleration
    X_tt = Xd_tt + Kv * delt_X_t + Kp * delt_X + ion * 0.01 * 10**4 * E_ * Dt

    # Control torque
    tau = xMc @ (pinv(Jk) @ ((X_tt - dt_Jk @ dq).T)) + vFc.reshape(3,1)

    # Force calculation
    Taw = Jk.T @ S_
    pinv_Taw = pinv(Taw) @ tau
    null_Taw = null_space(Taw)

    # Force saturation
    Tmax = 500
    T = pinv_Taw
    c = 1

    while np.any(T < 0) and np.all(T < Tmax):
        T += c * null_Taw
        if np.all(null_Taw <= 0):
            c = -1
        else:
            c = 1

    # Output torque
    Tau = r * T
    F = 0

    # Saturation limits
    maxtau = np.inf + 400 * r
    Tau[np.abs(Tau) > maxtau] = np.sign(Tau[np.abs(Tau) > maxtau]) * maxtau

    # Update time history
    if t > Args["t_"]:
        T = np.concatenate(([t], T, X[:2, 0], X[2:4, 0]))
        TT = np.concatenate((TT, [T]), axis=1)

    Args["t_"] = t

    #print(Tau)
    return Tau, F

def dyn_0(x, Args):
    # Extract variables J1, J2, J3, m1, m2, m3, l1, l2, l3, g = Args
    g = Args["g"]
    J1 = Args["J1"]
    J2 = Args["J2"]
    J3 = Args["J3"]
    m1 = Args["m1"]
    m2 = Args["m2"]
    m3 = Args["m3"]
    l1 = Args["l1"]
    l2 = Args["l2"]
    l3 = Args["l3"]
   
    q1, q2, q3, dq1, dq2, dq3 = x
    
    # Inertia matrix
    _ , xMc = Mass_matrix(x ,Args)

    # Coriolis and gravity forces
    vFc = np.array([
        -(1/2) * m2 * l1 * l2 * dq2**2 * np.sin(q2) - m3 * l1 * l2 * dq2**2 * np.sin(q2) - (1/2) * m3 * l1 * l3 * dq2**2 * np.sin(q2 + q3) - m2 * l1 * dq1 * l2 * np.sin(q2) * dq2 - 2 * m3 * l1 * dq1 * l2 * np.sin(q2) * dq2 - m3 * l1 * dq1 * l3 * np.sin(q2 + q3) * dq2 - m3 * l1 * l3 * dq3 * np.sin(q2 + q3) * dq2 - m3 * l2 * dq2 * l3 * np.sin(q3) * dq3 - m3 * l1 * dq1 * l3 * np.sin(q2 + q3) * dq3 - m3 * l2 * dq1 * l3 * np.sin(q3) * dq3 - (1/2) * m3 * l1 * l3 * dq3**2 * np.sin(q2 + q3) - (1/2) * m3 * l2 * l3 * dq3**2 * np.sin(q3) + (1/2) * m3 * g * l3 * np.cos(q1 + q2 + q3) + (1/2) * m1 * g * l1 * np.cos(q1) + m2 * g * l1 * np.cos(q1) + m3 * g * l1 * np.cos(q1) + (1/2) * m2 * g * l2 * np.cos(q1 + q2) + m3 * g * l2 * np.cos(q1 + q2),
        -m3 * l2 * dq2 * l3 * np.sin(q3) * dq3 + (1/2) * np.sin(q2) * dq1**2 * l1 * l2 * m2 + np.sin(q2) * dq1**2 * l1 * l2 * m3 + (1/2) * dq1**2 * np.sin(q2 + q3) * l1 * l3 * m3 - m3 * l2 * dq1 * l3 * np.sin(q3) * dq3 - (1/2) * m3 * l2 * l3 * dq3**2 * np.sin(q3) + (1/2) * m3 * g * l3 * np.cos(q1 + q2 + q3) + (1/2) * m2 * g * l2 * np.cos(q1 + q2) + m3 * g * l2 * np.cos(q1 + q2),
        (1/2) * m3 * l3 * (dq2**2 * np.sin(q3) * l2 + 2 * dq2 * dq1 * np.sin(q3) * l2 + dq1**2 * np.sin(q2 + q3) * l1 + dq1**2 * np.sin(q3) * l2 + g * np.cos(q1 + q2 + q3))
    ])

    return xMc, vFc

def planning(t, lj, q, dq, X, Dt, Args):
    
    # Extract variables l1, l2, l3, ivp, alpha, Xi, Xf = Args
    l1 = Args["l1"]
    l2 = Args["l2"]
    l3 = Args["l3"]
    ivp = Args["ivp"]
    Xi = Args["Xi"]
    Xf = Args["Xf"]
    alpha = Args["alpha"]

    # Calculate segment lengths for each joint
    l11, l12, l13 = calculateSegments(lj[0], l1, l2, l3)
    l21, l22, l23 = calculateSegments(lj[1], l1, l2, l3)
    l31, l32, l33 = calculateSegments(lj[2], l1, l2, l3)
    l41, l42, l43 = calculateSegments(lj[3], l1, l2, l3)

    # Trajectory planning
    Vi = np.zeros(2)
    Vf = np.zeros(2)
    tf = 1.0

    a, b, c, d = computeTrajectoryCoefficients(Xi, Xf, Vi, Vf, tf)

    if t <= tf:
        Xc, Vc, Ac = evaluateTrajectory(a, b, c, d, t)
    else:
        Xc, Vc, Ac = evaluateTrajectory(a, b, c, d, tf)

    X3d = Xc
    X2d = Xc + l3 * np.array([np.cos(alpha), np.sin(alpha)])

    V3 = Vc
    V2 = Vc
    A3 = Ac
    A2 = Ac

    # Inverse kinematics
    qc = invKIN(Xc + l3 * np.array([np.cos(alpha), np.sin(alpha)]), Xc ,Args)

    # Jacobians and their time derivatives
    dqc = computeJointVelocities(l1, l2, l3, qc, Vc)
    ddqc = computeJointAccelerations(l1, l2, l3, qc, dqc, Ac)

    # Compute positions, velocities, and accelerations
    Jk, dt_Jk, Xd = computeJacobians(l11, l12, l13, l21, l22, l23, l31, l32, l33, l41, l42, l43, qc, dqc)

    Vd = Jk @ dqc
    Ad = Jk @ ddqc + dt_Jk @ dqc
    return Xd, Vd, Ad

def calculateSegments(lj, l1, l2, l3):
    # Calculates segment lengths based on joint positions
    if lj <= l1:
        l1_seg = lj
        l2_seg = 0
        l3_seg = 0
    elif lj <= l1 + l2:
        l1_seg = l1
        l2_seg = lj - l1
        l3_seg = 0
    elif lj <= l1 + l2 + l3:
        l1_seg = l1
        l2_seg = l2
        l3_seg = lj - (l1 + l2)
    else:
        l1_seg = 0
        l2_seg = 0
        l3_seg = 0
    return l1_seg, l2_seg, l3_seg

def computeTrajectoryCoefficients(Xi, Xf, Vi, Vf, tf):
    # Computes trajectory coefficients for cubic polynomial
    a = -(2 / tf**3) * (Xf - Xi) + (1 / tf**2) * (Vf + Vi)
    b = (3 / tf**2) * (Xf - Xi) - (2 / tf) * Vi - (1 / tf) * Vf
    c = Vi
    d = Xi
    return a, b, c, d

def evaluateTrajectory(a, b, c, d, t):
    # Evaluates the trajectory at time t
    Xc = a * t**3 + b * t**2 + c * t + d
    Vc = 3 * a * t**2 + 2 * b * t + c
    Ac = 6 * a * t + 2 * b
    return Xc, Vc, Ac

def computeJointVelocities(l1, l2, l3, qc, Vc):
    # Computes joint velocities using Jacobian pseudoinverse
    Jkc3, _, _ = jak(l1, l2, l3, qc, np.zeros(3))
    Jkc2, _, _ = jak(l1, l2, 0, qc, np.zeros(3))
    dqc = (0.5 * pinv(Jkc3) + 0.5 * pinv(Jkc2)) @ Vc
    return dqc

def computeJointAccelerations(l1, l2, l3, qc, dqc, Ac):
    # Computes joint accelerations using Jacobian and its time derivative
    Jkc3, dt_Jkc3, _ = jak(l1, l2, l3, qc, dqc)
    Jkc2, dt_Jkc2, _ = jak(l1, l2, 0, qc, dqc)

    # Compute joint accelerations using Jacobian pseudoinverse
    ddqc = (0.5 * pinv(Jkc3) + 0.5 * pinv(Jkc2)) @ (Ac - (0.5 * dt_Jkc3 + 0.5 * dt_Jkc2) @ dqc)

    return ddqc

def computeJacobians(l11, l12, l13, l21, l22, l23, l31, l32, l33, l41, l42, l43, qc, dqc):
    # Computes Jacobians and their time derivatives
    Jk1, dt_Jk1, X1 = jak(l11, l12, l13, qc, dqc)
    Jk2, dt_Jk2, X2 = jak(l21, l22, l23, qc, dqc)
    Jk3, dt_Jk3, X3 = jak(l31, l32, l33, qc, dqc)
    Jk4, dt_Jk4, X4 = jak(l41, l42, l43, qc, dqc)

    Jk = np.vstack((Jk1, Jk2, Jk3, Jk4))
    dt_Jk = np.vstack((dt_Jk1, dt_Jk2, dt_Jk3, dt_Jk4))
    
    #--------
    Xd = np.vstack((X1, X2, X3, X4)).reshape(8) 
    #--------

    return Jk, dt_Jk, Xd