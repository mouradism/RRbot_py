import numpy as np
from scipy.linalg import pinv, null_space
from jak import jak
from invKIN import invKIN
from v_0.cables_v0 import cables

def controle(t, x, args):
    # Extract global variables from args
    l1, l2, l3 = args['l1'], args['l2'], args['l3']
    Lb, r, La, t_, E_, TT = args['Lb'], args['r'], args['La'], args['t_'], args['E_'], args['TT']
    lj, XL = args['lj'], args['XL']

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
        T = T + c * null_Taw
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
        T = np.array([t, *T, *X[:2], *X[2:4]])
        TT = np.column_stack((TT, T))

    args['t_'] = t
    args['E_'] = E_
    args['TT'] = TT

    return Tau, F

# Example of how to call the function
args = {
    'l1': 1.0,
    'l2': 1.0,
    'l3': 1.0,
    'Lb': 1.0,
    'r': 1.0,
    'La': 1.0,
    't_': 0.0,
    'E_': 0.0,
    'TT': np.zeros((5, 0)),
    'lj': 1.0,
    'XL': 1.0
}

t = 0.1
x = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
Tau, F = controle(t, x, args)
print("Tau:", Tau)
print("F:", F)
