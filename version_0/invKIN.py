import numpy as np

def invKIN(X2, X3, l1, l2):
    # Calculate the cosine of the angle q2
    cph2r = (X2[0]**2 + X2[1]**2 - l1**2 - l2**2) / (2 * l1 * l2)
    
    # Calculate q2 using the arctangent function
    q2 = np.arctan2(np.sqrt(1 - cph2r**2), cph2r)
    
    # Calculate q1 using the arctangent function
    q1 = np.arctan2(X2[1] * (l1 + l2 * cph2r) - X2[0] * l2 * np.sin(q2),
                    X2[0] * (l1 + l2 * cph2r) + X2[1] * l2 * np.sin(q2))
    
    # Calculate q3 using the arctangent function
    q3 = np.arctan2(X2[1] - X3[1], X2[0] - X3[0]) - (q1 + q2) - np.pi
    
    # Normalize the angles to be within the range [-pi, pi]
    q = np.arctan2(np.sin([q1, q2, q3]), np.cos([q1, q2, q3]))
    
    return q
