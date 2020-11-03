import numpy as np
import datagen
from datagen import sigmaEllipse2D
import matplotlib
import matplotlib.pyplot as plt


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

    r = r_pred
    Cr = Cr_pred

    p = p_pred
    Cp = Cp_pred

    # Go through all measurements
    for y in z:
        CI, CII, M = get_auxiliary_variables(p, Cp, Ch)

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
        Ybar = F@Cy.flatten()
        CpY = Cp@M.T
        CY = F@np.kron(Cy, Cy)@(F + Ftilde).T

        # Update shape
        p = p + CpY@np.linalg.inv(CY)@(Y-Ybar)
        Cp = Cp - CpY@np.linalg.inv(CY)@CpY.T

        Cp = (Cp+Cp.T)/2

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

    return CI, CII, M



if __name__ == '__main__':
    # Covariance of multiplicative noise
    Ch = np.diag([1/4, 1/4])
    # Covariance of measurement noise
    Cv = np.diag([10, 10])
    # Covariance for the process noise of the kinematic state
    Cwr = np.diag([1, 1, 0.1, 0.1])
    # Covariance for the process noise of the shape parameters
    Cwp = np.diag([0.5, 10, 10])

    # Prior
    r = np.array([0, 0, 0, 0])
    p = np.array([-np.pi/3, 3, 3])

    Cr = np.diag([1, 1, 1, 1])
    Cp = np.diag([0.2, 400, 400])

    Ar = np.array([[1, 0, 10, 0], 
                   [0, 1, 0,  10],
                   [0, 0, 1,  0],
                   [0, 0, 0,  1]])
    Ap = np.eye(3)

    states, extents, Z = datagen.generate_scenario()

    K = 100

    estimates = []

    R = lambda x: np.array([[np.cos(x),-np.sin(x)],
                            [np.sin(x), np.cos(x)]])

    # plot ground truth
    plt.scatter([state[0] for state in states], [state[1]for state in states])

    for k in range(K):
        print(k)
        # Predict
        r, Cr, p, Cp = predict(r, Cr, p, Cp, Ar, Ap, Cwr, Cwp)

        # Update
        r, Cr, p, Cp = update(Z[k], r, Cr, p, Cp, Cv, Ch)

        estimates.append((r, p))
        plt.scatter([r[0]], [r[1]])

        ext = np.diag([p[1], p[2]])
        ext = R(p[0])@ext@R(p[0]).T
        xy = sigmaEllipse2D(r[0:2], ext)
        plt.scatter(Z[k][:,0], Z[k][:,1])
        # plt.plot(xy[0,:], xy[1,:])

    plt.show()

