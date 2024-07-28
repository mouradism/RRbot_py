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

# Extract variables
F3_o = Args["F3_o"]
F2_o = Args["F2_o"]
t_ = Args["t_"]
E_ = Args["E_"]
TT = Args["TT"]
FF = Args["FF"]
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
La = Args["La"]
Lb = Args["Lb"]
r = Args["r"]
ivp = Args["ivp"]
Xi = Args["Xi"]
Xf = Args["Xf"]
alpha = Args["alpha"]
lj = Args["lj"]
XL = Args["XL"]

# Display extracted variables
print("F3_o:", F3_o)
print("F2_o:", F2_o)
print("t_:", t_)
print("E_:", E_)
print("TT:", TT)
print("FF:", FF)
print("g:", g)
print("J1:", J1)
print("J2:", J2)
print("J3:", J3)
print("m1:", m1)
print("m2:", m2)
print("m3:", m3)
print("l1:", l1)
print("l2:", l2)
print("l3:", l3)
print("La:", La)
print("Lb:", Lb)
print("r:", r)
print("ivp:", ivp)
print("Xi:", Xi)
print("Xf:", Xf)
print("alpha:", alpha)
print("lj:", lj)
print("XL:", XL)
