import numpy as np

def configr(l1, l2, l3, La, Lb):
    # Define lj
    lj = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
        [1, 1, 0]
    ]).dot(np.array([l1, l2, l3]))

    # Check condition//not required
    if np.any(lj > l1 + l2 + l3):
        return None, None

    # Define XL
    XL = np.array([
        [-Lb / 2, -Lb / 2, +Lb / 2, +Lb / 2],
        [-La, 0, 0, -La]
    ])

    return lj, XL