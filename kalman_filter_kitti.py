import copy

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal

import utils
from KalmanFilter import KalmanFilter
import pdb


def kalman_filter_kitti(p0, q, r, n, k, num_frames, measurements, F, H, point_index, cone):
    # Initialize KalmanFilter object and define initial state.
    #pdb.set_trace()
    start_index = point_index - n + 1  # XXX
    num_frames = num_frames - start_index
    m_dim, n_dim = H.shape
    k_filter = KalmanFilter(F, H, P0=p0, Q=q, R0=r)
    covariances = np.empty((num_frames, n_dim, n_dim))  # initialize array and include first P
    covariances[0] = k_filter.P

    num_cycles = num_frames - (n + k) + 1
    num_predicted = num_cycles  # Total number of predicted points.
    pred_trajectory = np.empty((num_predicted, n_dim))
    pred_covariances = np.empty((num_predicted, n_dim, n_dim))
    init_trajectory = []
    init_cov = []
    x_hat_previous = None

    corr_num = n

    s = start_index  # incremented by 1 every iteration
    m = start_index  # incremented by 1 every iteration

    while s < start_index + num_cycles:

        k_filter = KalmanFilter(F, H, P0=p0, Q=q, R0=r)
        # State initialization.
        x0, y0 = measurements[s]
        v0x = (measurements[s + 1, 0] - x0) * 10
        v0y = (measurements[s + 1, 1] - y0) * 10
        if n_dim == 6:
            x_hat_previous = np.array([x0, v0x, y0, v0y, 0, 0])
        elif n_dim == 4:
            x_hat_previous = np.array([x0, v0x, y0, v0y])

        # Take the first n frames and fit the kalman filter on them.
        for j in range(m, m + corr_num):
            z_k = measurements[j]
            x_hat_new = k_filter.fit(x_hat_previous, z_k)  # predict and correct
            x_hat_previous = x_hat_new

        # Take k frames and predict the position without correction.
        for j in range(m + corr_num, m + corr_num + k):
            x_hat_new = k_filter.predict(x_hat_previous)
            if not cone or (cone and s == start_index):
                #print("\tappend to init_cov/traj")  # , x_hat_new
                init_trajectory.append(x_hat_new)
                init_cov.append(k_filter.P)
            x_hat_previous = x_hat_new

        # Add only the last point to the predicted trajectory
        #print("add in pred_trajectory[{}]".format(s - start_index))
        pred_trajectory[s - start_index] = x_hat_previous
        pred_covariances[s - start_index] = k_filter.P

        s = s + 1
        m = m + 1


    init_trajectory = np.array(init_trajectory)
    pred_trajectory_total = init_trajectory[:, [0, 2]]
    pred_trajectory_total = pred_trajectory_total.reshape(-1, 40, 2)


    return pred_trajectory_total