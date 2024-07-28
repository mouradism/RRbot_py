import numpy as np
from TETA import TETA
from jak import jak

def cables(lj, XL, q, dq, l1, l2, l3):
    def get_lengths(lj_value, l1, l2, l3):
        if lj_value <= l1:
            return lj_value, 0, 0
        elif lj_value > l1 and lj_value <= l1 + l2:
            return l1, lj_value - l1, 0
        elif lj_value > (l1 + l2) and lj_value <= l1 + l2 + l3:
            return l1, l2, lj_value - (l1 + l2)
        else:
            return None, None, None

    l11, l12, l13 = get_lengths(lj[0], l1, l2, l3)
    l21, l22, l23 = get_lengths(lj[1], l1, l2, l3)
    l31, l32, l33 = get_lengths(lj[2], l1, l2, l3)
    l41, l42, l43 = get_lengths(lj[3], l1, l2, l3)

    if None in [l11, l12, l13, l21, l22, l23, l31, l32, l33, l41, l42, l43]:
        return None, None, None, None

    Jk1, dt_Jk1, X1 = jak(l11, l12, l13, q, dq)
    Jk2, dt_Jk2, X2 = jak(l21, l22, l23, q, dq)
    Jk3, dt_Jk3, X3 = jak(l31, l32, l33, q, dq)
    Jk4, dt_Jk4, X4 = jak(l41, l42, l43, q, dq)

    S = np.zeros((2, 4))
    S[:, 0] = -np.array([np.cos(TETA(XL[0, 0], XL[1, 0], X1)), np.sin(TETA(XL[0, 0], XL[1, 0], X1))])
    S[:, 1] = -np.array([np.cos(TETA(XL[0, 1], XL[1, 1], X2)), np.sin(TETA(XL[0, 1], XL[1, 1], X2))])
    S[:, 2] = -np.array([np.cos(TETA(XL[0, 2], XL[1, 2], X3)), np.sin(TETA(XL[0, 2], XL[1, 2], X3))])
    S[:, 3] = -np.array([np.cos(TETA(XL[0, 3], XL[1, 3], X4)), np.sin(TETA(XL[0, 3], XL[1, 3], X4))])

    Jk = np.vstack([Jk1, Jk2, Jk3, Jk4])
    dt_Jk = np.vstack([dt_Jk1, dt_Jk2, dt_Jk3, dt_Jk4])
    X = np.vstack([X1, X2, X3, X4])

    S_ = np.block([
        [S[:, 0].reshape(2, 1), np.zeros((2, 3))],
        [np.zeros((2, 1)), S[:, 1].reshape(2, 1), np.zeros((2, 2))],
        [np.zeros((2, 2)), S[:, 2].reshape(2, 1), np.zeros((2, 1))],
        [np.zeros((2, 3)), S[:, 3].reshape(2, 1)]
    ])

    return X, Jk, dt_Jk, S_

"""
# Example usage
lj = np.array([1.0, 2.0, 3.0, 4.0])
XL = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
q = np.array([0.1, 0.2, 0.3])
dq = np.array([0.4, 0.5, 0.6])
l1, l2, l3 = 1.0, 1.0, 1.0

X, Jk, dt_Jk, S_ = cables(lj, XL, q, dq, l1, l2, l3)
print("X:", X)
print("Jk:", Jk)
print("dt_Jk:", dt_Jk)
print("S_:", S_)
"""