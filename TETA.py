import numpy as np

def TETA(x, y, X):
    teta = np.arctan2(y - X[1], x - X[0]) - np.pi
    return teta