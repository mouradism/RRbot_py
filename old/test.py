import numpy as np
from TETA import TETA
from Mass_matrix import Mass_matrix

#Example use case for  Mass_matrix.py---------------------
g = 9.8
Lb, La = 2.0, 2.0
r = 0.25

l1, l2, l3 = 0.5, 0.5, 0.3
m1, m2, m3 = 6, 4, 1

J1 =(m1*l1**2)/3
J2 =(m2*l2**2)/3
J3 =(m3*l3**2)/3
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ivp = [x, J1, J2, J3, m1, m2, m3, l1, l2, l3]
result = Mass_matrix(ivp)
print("Example use case for  Mass_matrix.py")
print("Mass Matrix:")
print(result)


# Example use case for TETA.py-----------------------------
# Reference point (robot's position)
X = [1.0, 1.0]

# Target point (x, y)
x = 4.0
y = 5.0

# Calculate the angle teta
teta = TETA(x, y, X)

print("Example use case for TETA.py")
print(f"The angle teta from the reference point {X} to the target point ({x}, {y}) is: {teta} radians")
