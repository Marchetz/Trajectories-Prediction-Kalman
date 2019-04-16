import matplotlib.pyplot as plt
import numpy as np


def get_object_measurements(tracklets, obj_id):
    list_coords = tracklets[obj_id]
    xy_coords = [(-y, x) for (x, y, z) in list_coords]
    xy_coords = np.array(xy_coords)
    return xy_coords  # ndarray with shape (num_frames,2)


def normalize(v):
    norm = np.linalg.norm(v)
    norm = 1 if norm == 0 else norm
    return v / norm


def compute_external_points(trajectory, covariances):
    # initialize empty arrays
    right_points = np.empty((len(trajectory), 2))
    left_points = np.empty((len(trajectory), 2))
    diff = []
    lr_vectors = []

    # calculate trajectory vectors as differences between consecutive points
    if len(trajectory) < 2:
        return
    for i in range(0, len(trajectory) - 1):
        diff.append(trajectory[i + 1] - trajectory[i])

    # calculate the right and left orthogonal vectors to trajectory vectors
    for d in diff:
        # d = [d[0], d[2]]
        array1 = np.array([-d[1], d[0]])
        array2 = np.array([d[1], -d[0]])
        if np.cross(array1, d) > 0:  # use cross product sign to distinguish right and left vectors
            lr_vectors.append((array2, array1))  # left vector
        else:
            lr_vectors.append((array1, array2))  # right vector
    lr_vectors.append(lr_vectors[-1])  # otherwise lr_vectors will be shorter

    # compute right and left points as displacements from trajectory points in right and left directions
    for i in range(len(trajectory)):
        pl = trajectory[i] + normalize(lr_vectors[i][0]) * 2 * np.sqrt(covariances[i].diagonal())
        pr = trajectory[i] + normalize(lr_vectors[i][1]) * 2 * np.sqrt(covariances[i].diagonal())
        right_points[i, :] = pr
        left_points[i, :] = pl

    return right_points, left_points


# construct a 2d gaussian around each point of the trajectory and plot it
def plot_gaussians(covariances, kalman_trajectory):
    for cov, point in zip(covariances, kalman_trajectory):
        px, py = point
        gauss_x, gauss_y = np.random.multivariate_normal((px, py), cov, 128).T
        plt.scatter(gauss_x, gauss_y, s=1, c="orange", alpha=0.4)


def plot_covariance_stripe(right_points, left_points):
    ax = plt.gca()
    ax.plot(right_points[:, 0], right_points[:, 1], color="black", linestyle='dashed', linewidth=1)
    ax.plot(left_points[:, 0], left_points[:, 1], color="red", linestyle='dashed', linewidth=1)
    x = np.append(right_points[:, 0], left_points[:, 0][::-1])
    y = np.append(right_points[:, 1], left_points[:, 1][::-1])
    p = plt.Polygon(np.c_[x, y], color="red", alpha=0.2)
    ax.add_patch(p)


def maximize():
    print('backend:', plt.get_backend() ) # ivan -> Qt5Agg)
    figmng = plt.get_current_fig_manager()
    # plt.switch_backend('TkAgg')

    try:
        figmng.window.showMaximized()  # ivan -> funziona solo questo
        print("window.showMaximized()")
        return
    except AttributeError:
        pass
    try:
        figmng.frame.Maximize(True)
        print("frame.Maximize(True)")
        return
    except AttributeError:
        pass
    try:
        figmng.resize(*figmng.window.maxsize())
        print("resize(*figmng.window.maxsize())")
        return
    except AttributeError:
        pass
    try:
        figmng.window.state('zoomed')
        print("window.state('zoomed')")
        return
    except AttributeError:
        print("Not able to show maximized plot.")


def check_drawable(ind, num_frames, n, k):
    if n - 1 <= ind < num_frames - k:  # XXX
        return True
    else:
        print("Error: cannot start a correction-prediction cycle in the selected point." \
              "\nPlease select another point.")
        choice = 'behind' if ind < n - 1 else 'forward'
        plt.title("Cannot draw from this point. Not enough room " + choice + ".")
        plt.draw()
        return False


def init_matrices(acceleration, dt=0.1):
    if acceleration:  # use acceleration
        # transition matrix  x  x' y  y' x'' y''
        F = np.array([[1, 1 * dt, 0, 0, 0.5 * dt * dt, 0],  # x
                      [0, 1, 0, 0, 1 * dt, 0],  # x'
                      [0, 0, 1, 1 * dt, 0, 0.5 * dt * dt],  # y
                      [0, 0, 0, 1, 0, 1 * dt],  # y'
                      [0, 0, 0, 0, 1, 0],  # x''
                      [0, 0, 0, 0, 0, 1]])  # y''

        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]])
    else:
        F = np.array([[1, 1 * dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 1 * dt],
                      [0, 0, 0, 1]])

        H = np.array([[1, 0, 0, 0],  # m x n     m = 2, n = 4 or 6
                      [0, 0, 1, 0]])
    return F, H
