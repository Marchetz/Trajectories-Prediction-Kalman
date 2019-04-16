import utils
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import cv2
import datetime
import json
import pdb
import parse_trajectories

name_test = str(datetime.datetime.now())[:19]
if not os.path.exists(name_test):
    os.makedirs(name_test)
path_complete = name_test + '/'


# dataset_trajectories.json: trajectories in pixel coordinate in image plane
json_data = open('dataset_trajectories_small.json')
data = json.load(json_data)


all_3d_tracks, drive_list, track_list = parse_trajectories.get_all_3d_tracks(False, draw_tracks=False, plot_3d=False)


# video '0006' in dataset_trajectories.json <-> video '0018' in Kitti
# frame pixels: selected manually from image 'scene.png'
pts_src = np.array([[327, 298], [1171, 334], [507, 212],[733, 212]])
# real measurements (meters)
pts_dst = np.array([[0, 0],[9.70, 0],[0, 21.90],[9.70, 21.90]])

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)
h_inverse = np.linalg.inv(h)

error = []
error_mean = 0
error_1s = 0
error_2s = 0
error_3s = 0
error_4s = 0
tracks_total = 0
acceleration = True

videos = ['0006']
index_kitti = ['0018']

for i in range(len(videos)):
    v = 'video_' + videos[i]
    dict = {}
    i_dict = 0
    for track in track_list:
        if track.drive_name[-9:-5] == index_kitti[i]:
            dict['object_' + str(i_dict)] = track.instance
            i_dict += 1
    index_frames = np.array(list(data[v].keys()))
    for f in index_frames:
        index_objects = np.array(list(data[v][f].keys()))
        for o in index_objects:
            number_frame = int(f[-6:])
            video = v[-4:]
            frame = f[-6:]
            past_pixel = np.array(data[v][f][o]['past'])
            present_pixel = np.array(data[v][f][o]['present'])
            future_pixel = np.array(data[v][f][o]['future'])

            if len(past_pixel) > 0 and len(future_pixel)>0:
                print('reconstruction!!')

                #remove points in frame borders: they are very noise!
                index_wrong_x = np.where(past_pixel[:, 0] >= 1232)[0]
                past_pixel = np.delete(past_pixel, index_wrong_x, 0)
                index_wrong_y = np.where(past_pixel[:, 1] >= 365)[0]
                past_pixel = np.delete(past_pixel, index_wrong_y, 0)
                index_wrong_x = np.where(past_pixel[:, 0] <= 0)[0]
                past_pixel = np.delete(past_pixel, index_wrong_x, 0)
                index_wrong_y = np.where(past_pixel[:, 1] <= 0)[0]
                past_pixel = np.delete(past_pixel, index_wrong_y, 0)

                index_wrong_x = np.where(future_pixel[:, 0] >= 1232)[0]
                future_pixel = np.delete(future_pixel, index_wrong_x, 0)
                index_wrong_y = np.where(future_pixel[:, 1] >= 365)[0]
                future_pixel = np.delete(future_pixel, index_wrong_y, 0)
                index_wrong_x = np.where(future_pixel[:, 0] <= 0)[0]
                future_pixel = np.delete(future_pixel, index_wrong_x, 0)
                index_wrong_y = np.where(future_pixel[:, 1] <= 0)[0]
                future_pixel = np.delete(future_pixel, index_wrong_y, 0)

                if len(past_pixel) > 0 and len(future_pixel) > 0:
                    error_mean_track = 0
                    path_save = path_complete + video + '/' + frame + '/'
                    if not os.path.exists(path_save):
                        os.makedirs(path_save)
                    past_pixel = np.append([present_pixel], past_pixel, axis=0)
                    past_pixel = np.flip(past_pixel, axis=0)

                    #to map pixel coordinates in world coordinates with homography
                    future_pixel = future_pixel.reshape(-1, 1, 2).astype(float)
                    past_pixel = past_pixel.reshape(-1, 1, 2).astype(float)
                    present_pixel = present_pixel.reshape(-1, 1, 2).astype(float)
                    future_meters = cv2.perspectiveTransform(future_pixel, h)
                    past_meters = cv2.perspectiveTransform(past_pixel, h)
                    present_meters = cv2.perspectiveTransform(present_pixel, h)
                    track_generate = np.concatenate((past_meters,future_meters)).squeeze()

                    # get ground-truth tracjectories from dataset Kitti
                    for track in track_list:
                        if track.drive_name[-9:-5] == index_kitti[i]:
                            if track.instance == dict[o]:
                                track_choice = track
                                frame_init = track_choice.frame_interval[0]
                    dim_past = len(past_pixel)
                    dim_future = len(future_pixel)

                    track_world = track_choice.points
                    present_index = number_frame - frame_init
                    present = track_world[present_index]

                    track_world_past = track_world[(present_index - dim_past+1):present_index]
                    track_world_present = track_world[present_index, :].reshape(1, 2)
                    track_world_future = track_world[present_index + 1:(present_index + 1 + dim_future)]
                    track_world = np.concatenate((track_world_past, track_world_present))
                    track_world = np.concatenate((track_world, track_world_future))

                    plt.plot([], [], ' ')
                    plt.plot(track_world[:, 0], track_world[:, 1], c='green', marker='o', markersize=3)
                    plt.plot(track_generate[:, 0], track_generate[:, 1], c='red', marker='o', markersize=3)
                    plt.axis('equal')
                    plt.legend(['Traiettoria: ', 'corretta', "generata dall'omografia" ])
                    plt.savefig(path_save + o + '_differentSdR.png')
                    plt.close()


                    #risolution of different system problem
                    track_generate = track_generate - present_meters
                    track_world = track_world - present

                    #estimate of rigid transform between correct trajectory and the one generated by homography
                    rig_transf = cv2.estimateRigidTransform(track_generate.squeeze(), track_world, fullAffine=False)

                    if rig_transf is not None:
                        rig_transf = np.append(rig_transf, [[0, 0, 1]], axis=0)
                        tracks_total += 1
                        rig_transf[0, 2] = 0
                        rig_transf[1, 2] = 0
                        a = rig_transf[0, 0]
                        b = rig_transf[0, 1]
                        c = rig_transf[1, 0]
                        d = rig_transf[1, 1]
                        scale_x = np.sign(a) * np.sqrt(np.power(a, 2) + np.power(b, 2))
                        scale_y = np.sign(d) * np.sqrt(np.power(c, 2) + np.power(d, 2))
                        rig_transf[0, 0] = rig_transf[0, 0] / scale_x
                        rig_transf[0, 1] = rig_transf[0, 1] / scale_x
                        rig_transf[1, 0] = rig_transf[1, 0] / scale_y
                        rig_transf[1, 1] = rig_transf[1, 1] / scale_y

                        track_generate = track_generate.reshape(-1, 1, 2).astype(float)
                        track_generate_RT = cv2.transform(track_generate, rig_transf).squeeze()

                        plt.plot([], [], ' ')
                        plt.plot(track_world[:, 0], track_world[:, 1], c='green', marker='o', markersize=3)
                        plt.plot(track_generate_RT[:, 0], track_generate_RT[:, 1], c='red', marker='o', markersize=3)

                        plt.axis('equal')
                        plt.legend(['Traiettoria: ', 'corretta', "generata dall'omografia" ])
                        plt.savefig(path_save + o + '_rebuild.png')
                        #plt.close()
                        for i_error in range(len(track_world)):
                            error_mean_track += np.linalg.norm(track_world[i_error] - track_generate_RT[i_error,:2])
                        error_mean_track = error_mean_track / len(track_world)
                        error_mean += error_mean_track
                        error.append(error_mean_track)

mean_error_mean = error_mean / tracks_total
var_error_mean = np.var(error)
min_error_mean = np.min(error)
max_error_mean = np.max(error)

pdb.set_trace()
num_bins = 20
plt.hist(error, num_bins, facecolor='blue', alpha=0.5)
plt.savefig(path_complete + 'hist.png')


file = open(path_complete + "errors_allTracks.txt", "w")
file.write("total tracks " + str(tracks_total) + '\n')
file.write("error mean: " + str(mean_error_mean) + '\n')
file.write("error var: " + str(var_error_mean) + '\n')
file.write("error min: " + str(min_error_mean) + '\n')
file.write("error max: " + str(max_error_mean) + '\n')
file.close()

