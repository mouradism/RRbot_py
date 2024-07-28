import numpy as np
from TETA import TETA
from jak import jak

def cables(lj, XL, q, dq, l1, l2, l3):
    # Validate input lengths
    if any(lj > l1 + l2 + l3):
        raise ValueError('One or more lengths exceed the total combined length.')

    # Helper function to distribute lengths
    def distribute_length(l, l1, l2):
        if l <= l1:
            return l, 0, 0
        elif l <= l1 + l2:
            return l1, l - l1, 0
        else:
            return l1, l2, l - (l1 + l2)

    # Distribute lengths
    l11, l12, l13 = distribute_length(lj[0], l1, l2)
    l21, l22, l23 = distribute_length(lj[1], l1, l2)
    l31, l32, l33 = distribute_length(lj[2], l1, l2)
    l41, l42, l43 = distribute_length(lj[3], l1, l2)

    # Calculate Jacobians and other values
    Jk1, dt_Jk1, X1 = jak(l11, l12, l13, q, dq)
    Jk2, dt_Jk2, X2 = jak(l21, l22, l23, q, dq)
    Jk3, dt_Jk3, X3 = jak(l31, l32, l33, q, dq)
    Jk4, dt_Jk4, X4 = jak(l41, l42, l43, q, dq)

    # Calculate S matrix
    S = np.zeros((2, 4))
    for i in range(4):
        X_i = eval(f"X{i+1}")
        S[:, i] = -np.array([np.cos(TETA(XL[0, i], XL[1, i], X_i)),
                             np.sin(TETA(XL[0, i], XL[1, i], X_i))])

    # Outputs
    Jk = np.vstack([Jk1, Jk2, Jk3, Jk4])
    dt_Jk = np.vstack([dt_Jk1, dt_Jk2, dt_Jk3, dt_Jk4])
    X = np.vstack([X1, X2, X3, X4])

    # Construct S_ matrix
    S_ = np.block([
        [S[:, 0].reshape(2, 1), np.zeros((2, 3))],
        [np.zeros((2, 1)), S[:, 1].reshape(2, 1), np.zeros((2, 2))],
        [np.zeros((2, 2)), S[:, 2].reshape(2, 1), np.zeros((2, 1))],
        [np.zeros((2, 3)), S[:, 3].reshape(2, 1)]
    ])

    return X, Jk, dt_Jk, S_