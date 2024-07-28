import numpy as np
from Mass_matrix import mass_matrix

g = 9.8
Lb, La = 2.0, 2.0
r = 0.25

l1, l2, l3 = 0.5, 0.5, 0.3
m1, m2, m3 = 6, 4, 1

J1 =(m1*l1**2)/3
J2 =(m2*l2**2)/3
J3 =(m3*l3**2)/3
x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
result = mass_matrix(x, J1, J2, J3, m1, m2, m3, l1, l2, l3)

print("Mass Matrix:")
print(result)
