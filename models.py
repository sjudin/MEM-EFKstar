import numpy as np

class CTMeasmodel:
    """
        Coordinated turn measurement model in 2D
        Input:
            sigma (scalar): standard deviation of measurement noise
        Output:
            d (scalar): measurement dimension
            H (2x5): return an observation matrix
            R (2x2): measurement noise covariance
            h (2x1): return a measurement
        Note:
            This assumes the state x is on the following form: [Xpos; Ypos; ***; ***; omega] or [Xpos; Ypos; Zpos; ***; ***; omega]
            where omega is turn rate and either cartesian velocity (vx, vy) or polar velocity (v, heading) can be used.
    """
    def __init__(self, sigma, dim=2):
        if dim not in (2,3):
            raise ValueError('Dimension argument needs to be 2 for 2D or 3 for 3D, given currently as {}.'.format(dim))

        if dim == 2:
            self.d = 2
            self.H = lambda x: np.array([[1,0,0,0,0],[0,1,0,0,0]])
            self.R = sigma**2*np.eye(2)
            self.h = lambda x: np.dot(self.H(x),x)
        elif dim == 3:
            self.d = 3
            self.H = lambda x: np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])
            self.R = sigma**2*np.eye(3)
            self.h = lambda x: np.dot(self.H(x),x)

class CVMeasmodel:
    """
        Constant velocity measurement model.
        Input:
            sigma (scalar): standard deviation of measurement noise
        Output:
            d (scalar): measurement dimension
            H (2x4): return an observation matrix
            R (2x2): measurement noise covariance
            h (2x1): return a measurement
        Note:
            This assumes the state x is on the following form: [Xpos; Ypos; Xvel; Yvel] or [Xpos; Ypos; Zpos; Xvel; Yvel; Zvel]
    """
    def __init__(self, sigma, dim=2):
        if dim not in (2,3):
            raise ValueError('Dimension argument needs to be 2 for 2D or 3 for 3D, given currently as {}.'.format(dim))

        if dim == 2:
            self.d = 2
            self.H = lambda x: np.array([[1,0,0,0],[0,1,0,0]])
            self.R = sigma**2*np.eye(2)
            self.h = lambda x: np.dot(self.H(x),x)
        elif dim == 3:
            self.d = 3
            self.H = lambda x: np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
            self.R = sigma**2*np.eye(3)
            self.h = lambda x: np.dot(self.H(x),x)

class CVModel:
    """
        Constant velocity motion model.
        Input:
            T (scalar): sampling time
            sigma (scalar): standard deviation of motion noise
            dim (scalar (2 || 3): Denotes if the motion model has [x;y] or [x;y;z] components
        Output:
            d (scalar): object state dimension
            F (2x2): return a motion transition matrix
            Q (4x4): motion noise covariance
            f (4x1): return state prediction
        Note:
            This assumes the state x is on the following form:
            [Xpos; Ypos; Xvel; Yvel] or [Xpos; Ypos; Zpos; Xvel; Yvel; Zvel]
    """
    def __init__(self, T, sigma, dim=2):
        if dim not in (2,3):
                raise ValueError('Dimension argument needs to be 2 for 2D or 3 for 3D, '\
                'given currently as {}'.format(dim))

        self.dim = dim

        if dim == 2:
            self.d = 4
            self.F = lambda x: np.array([[1, 0, T, 0], \
                                         [0, 1, 0, T], \
                                         [0, 0, 1, 0], \
                                         [0, 0, 0, 1]])
            self.Q = sigma**2*np.array([[T**4/4, 0,      T**3/2, 0], \
                                        [0,      T**4/4, 0,      T**3/2], \
                                        [T**3/2, 0,      T**2, 0], \
                                        [0,      T**3/2, 0,    T**2]])
        elif dim == 3:
            self.d = 6
            self.F = lambda x: np.array([[1, 0, 0, T, 0, 0], \
                                         [0, 1, 0, 0, T, 0], \
                                         [0, 0, 1, 0, 0, T], \
                                         [0, 0, 0, 1, 0, 0], \
                                         [0, 0, 0, 0, 1, 0], \
                                         [0, 0, 0, 0, 0, 1]])
            self.Q = sigma**2*np.array([[T**4/4, 0,      0,       T**3/2,  0,       0], \
                                        [0,      T**4/4, 0,       0,       T**3/2,  0], \
                                        [0,      0,      T**4/4,  0,       0,       T**3/2], \
                                        [T**3/2, 0,      0,       T**2,    0,       0], \
                                        [0,      T**3/2, 0,       0,       T**2,    0], \
                                        [0,      0,      T**3/2,  0,       0,       T**2]])

        self.f = lambda x: np.dot(self.F(x),x)


class CTModel:
    """ 
        Coordinated turn motion model using polar velocity.
        Input:
            T (scalar): sampling time
            sigma_a (scalar): standard deviation of motion noise added to polar velocity
            sigma_w (scalar): standard deviation of motion noise added to turn rate
            dim (scalar (2 || 3): Denotes if the motion model has [x;y] or [x;y;z] components
        Output:
            d (scalar): object state dimension
            F (5x5): return a motion Jacobian matrix
            Q (5x5): motion noise covariance
            f (5x1): return state prediction
        Note:
            This assumes the state x is on the following form: [Xpos; Ypos; vel; phi; omega]
            where phi is heading and omega is turn rate. vel is the polar
            velocity
    """
    def __init__(self, T, sigma_a, sigma_w, dim):
        if dim not in (2,3):
                raise ValueError('Dimension argument needs to be 2 for 2D or 3 for 3D, '\
                'given currently as {}'.format(dim))

        self.T = T
        self.dim = dim

        if dim == 2:
            self.d = 5
            self.f = lambda x: x + np.array([T*x[2]*np.cos(x[3]), \
                                             T*x[2]*np.sin(x[3]), \
                                             0, \
                                             T*x[4], \
                                             0])
            self.F = lambda x: np.array([[1, 0, T*np.cos(x[3]), -T*x[2]*np.sin(x[3]), 0], \
                                         [0, 1, T*np.sin(x[3]),  T*x[2]*np.cos(x[3]), 0], \
                                         [0, 0, 1,               0,                   0], \
                                         [0, 0, 0,               1,                   T], \
                                         [0, 0, 0,               0,                   1]])
            self.G = np.array([[0,0], [0,0], [1,0], [0,0], [0,1]])
            self.Q = np.dot(np.dot(self.G,np.diag(np.array([sigma_a**2, sigma_w**2]))),self.G.T)

            self.M = lambda x: np.array([[np.cos(x[-1]*T), -np.sin(x[-1]*T)], \
                                         [np.sin(x[-1]*T), np.cos(x[-1]*T)]])

            # First derivative of M
            self.M1 = lambda x: np.array([[-np.sin(x[-1]*T), -np.cos(x[-1]*T)], \
                                         [np.cos(x[-1]*T), -np.sin(x[-1]*T)]])*T

            # Second derivative of M
            self.M2 = lambda x: np.array([[-np.cos(x[-1]*T), np.sin(x[-1]*T)], \
                                         [-np.sin(x[-1]*T), -np.cos(x[-1]*T)]])*T**2

        elif dim == 3:
            self.d = 6
            self.f = lambda x: x + np.array([T*x[3]*np.cos(x[4]), \
                                             T*x[3]*np.sin(x[4]), \
                                             0, \
                                             0, \
                                             T*x[5], \
                                             0])
            self.F = lambda x: np.array([[1, 0, 0, T*np.cos(x[4]), -T*x[3]*np.sin(x[4]), 0], \
                                         [0, 1, 0, T*np.sin(x[4]),  T*x[3]*np.cos(x[4]), 0], \
                                         [0, 0, 1, 0,               0,                   0], \
                                         [0, 0, 0, 1,               0,                   0], \
                                         [0, 0, 0, 0,               1,                   T], \
                                         [0, 0, 0, 0,               0,                   1]])

            self.G = np.array([[0,0], [0,0], [0,0], [1,0], [0,0], [0,1]])
            self.Q = np.dot(np.dot(self.G,np.diag(np.array([sigma_a**2, sigma_w**2]))),self.G.T)
            self.M = lambda x: np.array([[np.cos(x[-1]*T), -np.sin(x[-1]*T), 0], \
                                         [np.sin(x[-1]*T), np.cos(x[-1]*T), 0], \
                                         [0,               0,               1]])
            self.M1 = lambda x: np.array([[-np.sin(x[-1]*T), -np.cos(x[-1]*T), 0], \
                                         [np.cos(x[-1]*T), -np.sin(x[-1]*T), 0], \
                                         [0,               0,               0]])*T
            self.M2 = lambda x: np.array([[-np.cos(x[-1]*T), np.sin(x[-1]*T), 0], \
                                         [-np.sin(x[-1]*T), -np.cos(x[-1]*T), 0], \
                                         [0,               0,               0]])*T**2


