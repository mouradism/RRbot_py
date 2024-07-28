import numpy as np

Args = {
    "F3_o": np.array([0, 0]),
    "F2_o": np.array([0, 0]),
    "t_": 0,
    "E_": 0,
    "TT": [],
    "FF": [],
    "g": 9.8,
    "J1": None,
    "J2": None,
    "J3": None,
    "m1": 6,
    "m2": 4,
    "m3": 1,
    "l1": 0.5,
    "l2": 0.5,
    "l3": 0.3,
    "La": 2.0,
    "Lb": 2.0,
    "r": 0.25,
    "ivp": None,
    "Xi": np.array([-0.5, -0.4]),
    "Xf": None,
    "alpha": -0.0 * np.pi,
    "lj": None,
    "XL": None
}

def construct_S_(S, S_,Args):
    S_[:2, :1] = S[:, 0].reshape(2, 1)
    S_[2:4, 1:2] = S[:, 1].reshape(2, 1)
    S_[4:6, 2:3] = S[:, 2].reshape(2, 1)
    S_[6:8, 3:4] = S[:, 3].reshape(2, 1)
    Args["t_"] = 100

S = np.array([
    [0.96623494, -1.0, 0.9486833, -0.95577901],
    [0.25766265, -0.0, 0.31622777, 0.29408585]
])

S_ = np.zeros((8, 4))
construct_S_(S, S_ ,Args)

print("S_:\n", S_)
print("Args[t_]:\n", Args["t_"])
