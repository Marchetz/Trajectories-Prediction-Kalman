# coding: UTF-8
import numpy as np
from numpy.linalg import inv


class KalmanFilter:

    def __init__(self, F, H, P0, Q, R0):
        """
        :param n: state dimension (4 in our case)
        :param m: measurement dimension (2 in our case)
        :param P0: initial process covariance matrix
        :param Q: process error covariance matrix
        """
        self.m = H.shape[0]
        self.n = H.shape[1]
        self.F = F

        self.H = H

        self.K = np.zeros((self.n, self.m))

        self.P = P0

        self.R = R0

        self.Q = Q

    def fit(self, xhat_previous, z_k):
        """
        main iteration: we need the state estimate x_hat_minus k-1 at previous step and the current measurement z_k

        :param xhat_previous: previous a posteriori prediction
        :param z_k: current measurement: (tx,ty) tuple
        :return: new a posteriori prediction
        """

        # prediction
        xhat_k_minus = self.predict(xhat_previous)  # predict updates self.P (P_minus)
        P_minus = self.P

        # correction
        inv_HPHTR = inv(np.dot(np.dot(self.H, P_minus), self.H.T) + self.R)
        self.K = np.dot(np.dot(P_minus, self.H.T), inv_HPHTR)

        residual = z_k - np.dot(self.H, xhat_k_minus)
        xhat_k_new = xhat_k_minus + np.dot(self.K, residual)

        self.P = np.dot((np.eye(self.n) - np.dot(self.K, self.H)), P_minus)

        return xhat_k_new

    def predict(self, xhat_k_previous):
        xhat_k_minus = np.dot(self.F, xhat_k_previous)  # update previous state estimate with state transition matrix

        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # this is P minus

        return xhat_k_minus
