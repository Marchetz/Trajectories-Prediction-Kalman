# coding: UTF-8
import argparse
import copy
import sys
import matplotlib.pyplot as plt
import numpy as np
import utils
from XMLParser import XMLParser
from kalman_filter_kitti import kalman_filter_kitti
import pdb
import datetime
import os
import parse_trajectories
from numpy.core.umath_tests import inner1d


def ModHausdorffDist(A,B):

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat,axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat,axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return (MHD, FHD, RHD)


# Parse command line arguments
parser = argparse.ArgumentParser(
    description='Trajectory prediction on KITTI dataset with Kalman Filter',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

group = parser.add_mutually_exclusive_group(required=False)

group.add_argument('-s', '--save', action='store_true', help='save the qualitative results')
parser.add_argument('-p0', type=float, default=15, help='P0 diagonal value')
parser.add_argument('-q', type=float, default=0.03, help='Q diagonal value')
parser.add_argument('-r0', type=float, default=0.03, help='R0 diagonal value')
parser.add_argument('--past_len', type=int, default=5, help='past length')
parser.add_argument('--future_len', type=int, default=40, help='future length')
parser.add_argument('-a', '--acceleration', action='store_true', help='use acceleration')

args = parser.parse_args()


""" PARAMETERS """
save_plot = args.save
n = args.past_len
k = args.future_len
p0 = args.p0
q = args.q
r0 = args.r0
acceleration = args.acceleration
cone = False

""" END PARAMETERS """

# save folder
name_test = str(datetime.datetime.now())[:19] + 'past_' + str(n) + 'acc_' + str(acceleration)
if not os.path.exists(name_test):
    os.makedirs(name_test)
path_complete = name_test + '/'


# -------------------------------------- START SCRIPT -------------------------------------- #

# Initialize the matrices
F, H = utils.init_matrices(acceleration)
m_dim, n_dim = H.shape
Q = np.diag(np.full(n_dim, q))
R0 = np.diag(np.full(m_dim, r0))
P0 = np.diag(np.full(n_dim, p0))
point_index = 19

# load xml
list_xml = os.listdir('tracklets')

# initialize variable for results
tracks_total = 0
error_1s = 0
error_2s = 0
error_4s = 0
error_3s = 0
error_mean = 0
mhd = 0

# get trajectories in world coordinates
train = False
all_3d_tracks, drive_list, track_list = parse_trajectories.get_all_3d_tracks(train, draw_tracks=False, plot_3d=False)

for track in track_list:
    video = track.drive_name[-9:-5]
    vehicle = track.instance
    measurements = track.points
    num_frames = len(measurements)
    if not os.path.exists(path_complete + video):
        os.makedirs(path_complete + video)
    video_path = path_complete + video + '/'

    if n - 1 <= point_index < num_frames - k:
        print('compute prediction track')
        if not os.path.exists(video_path + vehicle):
            os.makedirs(video_path + vehicle)
        vehicle_path = video_path + vehicle + '/'
        pred_trajectory = kalman_filter_kitti(P0, Q, R0, n, k, num_frames, measurements, F, H, point_index,cone
                                              )

        for i_pred, index in zip(range(len(pred_trajectory)), range(point_index - n + 1, num_frames - n - k + 1)):

            tracks_total += 1
            past_index = measurements[index:index + n]
            pred_index = pred_trajectory[i_pred]
            gt_index = measurements[index + n:index + n + k]
            present = past_index[-1]

            # track translation: the origin represent the present
            past_index = past_index - present
            pred_index = pred_index - present
            gt_index = gt_index - present

            # horizon metrics
            if k > 10:
                error_1s = np.linalg.norm(pred_index[9] - gt_index[9])
            if k > 20:
                error_2s = np.linalg.norm(pred_index[19] - gt_index[19])
            if k > 30:
                error_3s = np.linalg.norm(pred_index[29] - gt_index[29])
            if k >= 40:
                error_4s = np.linalg.norm(pred_index[39] - gt_index[39])
            error_mean_track = 0
            for i_error in range(len(pred_index)):
                error_mean_track += np.linalg.norm(pred_index[i_error] - gt_index[i_error])
            error_mean += error_mean_track / len(pred_index)
            (MHD, FHD, RHD) = ModHausdorffDist(pred_index, gt_index)
            mhd += FHD

            if save_plot:
                plt.axis('equal')
                # plt.plot(measurements[:, 0], measurements[:, 1], c='gray', marker='o', markersize=1)
                plt.plot([], [], ' ')
                plt.plot(past_index[:, 0], past_index[:, 1], c='blue', marker='o', markersize=3)
                plt.plot(gt_index[:, 0], gt_index[:, 1], c='green', marker='o', markersize=3)
                plt.plot(pred_index[:, 0], pred_index[:, 1], c='red', linewidth=1, marker='o', markersize=3)
                plt.legend(['Traiettoria: ', 'passata', 'futura corretta', 'futura predetta'])
                plt.savefig(vehicle_path + str(i_pred) + '.png')
                plt.close()

    else:
        print('too short')

# results analysis: mean computation of different metrics
mean_1s_allTracks = error_1s / tracks_total
mean_2s_allTracks = error_2s / tracks_total
mean_3s_allTracks = error_3s / tracks_total
mean_4s_allTracks = error_4s / tracks_total
mean_error_mean = error_mean / tracks_total
mean_mhd = mhd / tracks_total

# save results in a file
file = open(path_complete + "errors_allTracks.txt", "w")
file.write("Acceleration? " + str(acceleration) + '\n')
file.write("dimension of past" + str(n) + '\n')
file.write("dimension of future" + str(k) + '\n')
file.write("total tracks " + str(tracks_total) + '\n')
file.write("error 1s: " + str(mean_1s_allTracks) + '\n')
file.write("error 2s: " + str(mean_2s_allTracks) + '\n')
file.write("error 3s: " + str(mean_3s_allTracks) + '\n')
file.write("error 4s: " + str(mean_4s_allTracks) + '\n')
file.write("error mean: " + str(mean_error_mean) + '\n')
file.write("error mhd: " + str(mean_mhd) + '\n')
file.close()