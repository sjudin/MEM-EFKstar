import models
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import scipy.linalg

def sigmaEllipse2D(mu, cov, level = 3, npoints = 32, endpoint=True):
    """
        Method that returns points for the ellipse that can be used for plotting
        input:
            level (int): Which level curve of the ellipse is to be plotted, default 3
            npoints: How many points are used to approximate the ellipse, default 32
    """

    phi = np.linspace(0, 2*np.pi, npoints, endpoint=endpoint)

    # Extract radii from covariance matrix
    sqrt_cov = scipy.linalg.sqrtm(cov)
    t = np.array([np.cos(phi), np.sin(phi)]).reshape((-1,), order='F')
    xy = np.tile(mu, npoints) + level*np.dot(np.kron(np.eye(npoints), sqrt_cov), t)

    return xy.reshape((2,-1), order='F')


def generate_kinematic_state(a_traj, w_traj):
    """
        Return a list of kinematic state vectors 
        corresponding to ground truth
    """

    motionmodel = models.CTModel(1, 100, 100, 2)
    states = [np.zeros(5)]

    
    for k in range(1,100):
        states.append(motionmodel.f(states[k-1]) + motionmodel.G@np.array([a_traj[k], w_traj[k]]))

    return states

def generate_extended_state(states, init_extent):
    """
        Return a list of extended states given a kinematic
        state list. The extents are represented by covariance matrices
    """
    extents = [init_extent]
    R = lambda x: np.array([[np.cos(x[-1]),-np.sin(x[-1])],
                            [np.sin(x[-1]), np.cos(x[-1])]])

    for i, state in enumerate(states, 1):
        extents.append(R(state)@extents[i-1]@R(state).T)

    return extents


def generate_measurements(states, extents):
    Z = []

    for mu, cov in zip(states, extents):
        no_meas = np.random.poisson(10, 1)[0]

        Z.append(np.random.multivariate_normal(mu[0:2], cov, no_meas))

    return Z

def generate_scenario():
    # Generate some ground-truth states
    a_traj = np.zeros(100)
    a_traj[0:5] = 0.5
    w_traj = np.zeros(100)
    w_traj[0:10] = 0.08
    w_traj[11:21] = -0.08

    states = generate_kinematic_state(a_traj, w_traj)

    # Generate extension states
    ext = np.array([[2, 0], [0, 1]])
    extents = generate_extended_state(states, ext)

    # Generate some measurements
    Z = generate_measurements(states, extents)

    return states, extents, Z

if __name__ == '__main__':
    pass
