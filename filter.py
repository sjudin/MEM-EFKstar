import numpy as np
import datagen
from datagen import sigmaEllipse2D
import matplotlib
import matplotlib.pyplot as plt
import sys
from copy import copy

DEBUG = False
DEBUG_get_auxiliary_variables = False


def predict(r, Cr, p, Cp, Ar, Ap, Cwr, Cwp):
    """
        Perform the prediction (time update) given transition matrices
        and current states
    """

    r_pred = Ar@r
    Cr_pred = Ar@Cr@Ar.T + Cwr

    p_pred = Ap@p
    Cp_pred = Ap@Cp@Ap.T + Cwp

    return r_pred, Cr_pred, p_pred, Cp_pred


def update(z, r_pred, Cr_pred, p_pred, Cp_pred, Cv, Ch):
    """
        Perform the measurement update given a set of measurements
        for the current timestep
    """
    # Measurement matrix
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    # Measurement covariance
    F = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
    Ftilde = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    r = copy(r_pred)
    Cr = copy(Cr_pred)

    p = copy(p_pred)
    Cp = copy(Cp_pred)

    # Go through all measurements
    for i, y in enumerate(z):
        CI, CII, M = get_auxiliary_variables(p, Cp, Ch)

        # Predicted measurement
        ybar = H@r

        # Calculate moments for the kinematic state update
        Cry = Cr@H.T
        Cy = H@Cr@H.T + CI+CII+Cv

        # Update kinematic estimate
        r = r + Cry@np.linalg.inv(Cy)@(y-ybar)
        Cr = Cr - Cry@np.linalg.inv(Cy)@Cry.T
        # Enforce symmetry of the covariance
        Cr = (Cr+Cr.T)/2

        # Construct pseudo-measurement
        Y = F@np.kron(y-ybar, y-ybar)
        # Calculate moment for the shape update
        Ybar = F@Cy.flatten('C')
        # Ybar = F@Cy.flatten('F')

        CpY = Cp@M.T
        # CpY = np.eye(3)
        CY = F@np.kron(Cy, Cy)@(F + Ftilde).T

        # Update shape
        p = p + CpY@np.linalg.inv(CY)@(Y-Ybar)
        Cp = Cp - CpY@np.linalg.inv(CY)@CpY.T


        # Enforce symmetry of the covariance
        Cp = (Cp+Cp.T)/2

        if DEBUG:
            print(f"M: \n{M}")
            print(f"Y: \n{Y}")
            print(f"Ybar: \n{Ybar}")
            print(f"CpY: \n{CpY}")
            print(f"CY: \n{CY}")
            print(f"p: \n{p}")
            print(f"Cp: \n{Cp}")
            print("\n\n\n")
        # if i > 5:
        #     sys.exit()

    return r, Cr, p, Cp
        

def get_auxiliary_variables(p, Cp, Ch):
    S = np.array([[np.cos(p[0]), -np.sin(p[0])], 
                  [np.sin(p[0]),  np.cos(p[0])]])@np.diag([p[1], p[2]])

    S1 = S[0,:]
    S2 = S[1,:]

    J1 = np.array([[-p[1]*np.sin(p[0]), np.cos(p[0]), 0],
                   [-p[2]*np.cos(p[0]), 0, -np.sin(p[0])]])

    J2 = np.array([[ p[1]*np.cos(p[0]), np.sin(p[0]), 0],
                   [-p[2]*np.sin(p[0]), 0, np.cos(p[0])]])

    CI = S@Ch@S.T
    CII = np.zeros((2,2))

    CII[0,0] = np.trace(Cp@J1.T@Ch@J1)
    CII[0,1] = np.trace(Cp@J2.T@Ch@J1)
    CII[1,0] = np.trace(Cp@J1.T@Ch@J2)
    CII[1,1] = np.trace(Cp@J2.T@Ch@J2)

    M = np.array([2*S1@Ch@J1, 2*S2@Ch@J2, S1@Ch@J2 + S2@Ch@J1])

    if DEBUG_get_auxiliary_variables:
        print(f"S1: \n{S1}")
        print(f"S2: \n{S2}")
        print(f"J1: \n{J1}")
        print(f"J2: \n{J2}")
        print(f"M: \n{M}")
        print("\n\n\n")

    return CI, CII, M



if __name__ == '__main__':
    # Covariance of multiplicative noise
    Ch = np.diag([1/4, 1/4])
    # Covariance of measurement noise
    Cv = np.diag([1, 1])
    # Covariance for the process noise of the kinematic state
    Cwr = np.diag([1, 1, 1, 1])
    # Covariance for the process noise of the shape parameters
    Cwp = np.diag([1, 1, 1])

    # Prior
    r = np.array([0, 0, 0, 0])
    p = np.array([-np.pi/3, 3, 3])

    Cr = np.diag([1, 1, 1, 1])
    Cp = np.diag([0.5, 10, 10])

    Ar = np.array([[1, 0, 1, 0], 
                   [0, 1, 0, 1],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    Ap = np.eye(3)

    states, extents, Z = datagen.generate_scenario()

    K = 100

    estimates = []

    R = lambda x: np.array([[np.cos(x),-np.sin(x)],
                            [np.sin(x), np.cos(x)]])

    # plot ground truth
    plt.scatter([state[0] for state in states], [state[1]for state in states])

    for k in range(K):

        # Update
        r, Cr, p, Cp = update(Z[k], r, Cr, p, Cp, Cv, Ch)

        estimates.append((r, p))


        ext = np.abs(np.diag([p[1], p[2]]))
        ext = R(p[0])@ext@R(p[0]).T
        xy = sigmaEllipse2D(r[0:2], ext)
        # plt.scatter(Z[k][:,0], Z[k][:,1])
        # plt.scatter([r[0]], [r[1]])
        print(f"alpha: {p[0]}, l1: {p[1]}, l2: {p[2]}")
        plt.plot(xy[0,:], xy[1,:])
        # gt_ext = sigmaEllipse2D(np.array([states[k][0], states[k][1]]), extents[k])
        # plt.plot(gt_ext[0,:], gt_ext[1,:])
        plt.pause(0.1)

        # Predict
        r, Cr, p, Cp = predict(r, Cr, p, Cp, Ar, Ap, Cwr, Cwp)

    plt.axis('equal')
    plt.show()

