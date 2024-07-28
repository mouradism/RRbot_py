import numpy as np
from jak import jak
from TETA import TETA
from scipy.linalg import block_diag

def distribute_length(l, l1, l2):
    if l <= l1:
        return l, 0, 0
    elif l <= l1 + l2:
        return l1, l - l1, 0
    else:
        return l1, l2, l - (l1 + l2)

def cables( q, dq, Args):
    l1, l2, l3 = Args["l1"], Args["l2"], Args["l3"]
    lj, XL = Args["lj"], Args["XL"]
    
    if any(l > l1 + l2 + l3 for l in lj):
        raise ValueError('One or more lengths exceed the total combined length.')

    l11, l12, l13 = distribute_length(lj[0], l1, l2)
    l21, l22, l23 = distribute_length(lj[1], l1, l2)
    l31, l32, l33 = distribute_length(lj[2], l1, l2)
    l41, l42, l43 = distribute_length(lj[3], l1, l2)

    Jk1, dt_Jk1, X1 = jak(l11, l12, l13, q, dq)
    Jk2, dt_Jk2, X2 = jak(l21, l22, l23, q, dq)
    Jk3, dt_Jk3, X3 = jak(l31, l32, l33, q, dq)
    Jk4, dt_Jk4, X4 = jak(l41, l42, l43, q, dq)

    S = np.zeros((2, 4))
    for i in range(4):
        X_eval = eval(f'X{i+1}')
        S[:, i] = -np.array([np.cos(TETA(XL[0, i], XL[1, i], X_eval)),
                             np.sin(TETA(XL[0, i], XL[1, i], X_eval))])

    Jk = np.vstack([Jk1, Jk2, Jk3, Jk4])
    dt_Jk = np.vstack([dt_Jk1, dt_Jk2, dt_Jk3, dt_Jk4])
    X = np.vstack([X1, X2, X3, X4]).reshape(1, 8)
    S_ = block_diag([S[:,0].T], [S[:,1].T], [S[:,2].T], [S[:,3].T]).T

    return X, Jk, dt_Jk, S_