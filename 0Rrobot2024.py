
import numpy as np
from invKIN import invKIN

# Dictionary to store global variables
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

# Calculate additional parameters
Args["J1"] = (Args["m1"] * Args["l1"] ** 2) / 3
Args["J2"] = (Args["m2"] * Args["l2"] ** 2) / 3
Args["J3"] = (Args["m3"] * Args["l3"] ** 2) / 3
Args["Xf"] = Args["Xi"] + np.array([0.40, -0.40])

# Link lengths
Args["lj"] = np.array([[1, 1, 1], [0.5, 0, 0], [1, 1, 0], [1, 0.5, 0]]) @ np.array([Args["l1"], Args["l2"], Args["l3"]])
if np.any(Args["lj"] > (Args["l1"] + Args["l2"] + Args["l3"])):
    raise ValueError("Link lengths exceed the sum of individual lengths")

Args["XL"] = np.array([[Args["Lb"] / 2, -Args["Lb"] / 2, Args["Lb"] / 2, -Args["Lb"] / 2], [0, 0, 0, 0]])

# Inverse Kinematics
q = invKIN(Args["Xi"] + Args["l3"] * np.array([np.cos(Args["alpha"]), np.sin(Args["alpha"])]), Args["Xi"],Args)
print(q)
# Initial conditions
dq1, dq2, dq3 = 0, 0, 0
Args["ivp"] = np.array([q[0], q[1], q[2], dq1, dq2, dq3])
