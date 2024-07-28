import numpy as np

def jak(l1, l2, l3, q, dq):
    Jk = np.array([
        [-l1*np.sin(q[0])-l2*np.sin(q[0]+q[1])-l3*np.sin(q[0]+q[1]+q[2]), -l2*np.sin(q[0]+q[1])-l3*np.sin(q[0]+q[1]+q[2]), -l3*np.sin(q[0]+q[1]+q[2])],
        [ l1*np.cos(q[0])+l2*np.cos(q[0]+q[1])+l3*np.cos(q[0]+q[1]+q[2]),  l2*np.cos(q[0]+q[1])+l3*np.cos(q[0]+q[1]+q[2]),  l3*np.cos(q[0]+q[1]+q[2])]
    ])

    dt_Jk = np.array([
        [-l1*np.cos(q[0])*dq[0]-l2*np.cos(q[0]+q[1])*(dq[0]+dq[1])-l3*np.cos(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2]), -l2*np.cos(q[0]+q[1])*(dq[0]+dq[1])-l3*np.cos(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2]), -l3*np.cos(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])],
        [-l1*np.sin(q[0])*dq[0]-l2*np.sin(q[0]+q[1])*(dq[0]+dq[1])-l3*np.sin(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2]), -l2*np.sin(q[0]+q[1])*(dq[0]+dq[1])-l3*np.sin(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2]), -l3*np.sin(q[0]+q[1]+q[2])*(dq[0]+dq[1]+dq[2])]
    ])

    X = np.array([
        l1*np.cos(q[0])+l2*np.cos(q[0]+q[1])+l3*np.cos(q[0]+q[1]+q[2]),
        l1*np.sin(q[0])+l2*np.sin(q[0]+q[1])+l3*np.sin(q[0]+q[1]+q[2])
    ])

    return Jk, dt_Jk, X
