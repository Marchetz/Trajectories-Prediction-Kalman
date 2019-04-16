# coding: UTF-8
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from XMLParser import XMLParser
from math import cos, sin, pi, log, tan
import pyproj
from mpl_toolkits import mplot3d


def lat_to_scale(lat):
    """ compute mercator scale from latitude """
    scale = cos(lat*pi/180.0)
    return scale


def latlon_to_mercator(lat, lon, scale):
    er = 6378137
    mx = scale * lon * pi * er / 180.0
    my = scale * er * log(tan((90.0 + lat) * pi / 360.0))
    return mx, my


def convert_oxts_to_pose(oxts):
    # python porting of the matlab version in the devkit (all variable names are kept the same as the original version)
    scale = lat_to_scale(oxts[0][0])
    pose = []
    Tr_0_inv = None

    for oxt in oxts:
        t1, t2 = latlon_to_mercator(oxt[0], oxt[1], scale)
        t3 = oxt[2]
        t = [t1, t2, t3]

        rx = oxt[3]
        ry = oxt[4]
        rz = oxt[5]
        Rx = np.array([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]])
        Ry = np.array([[cos(ry), 0, sin(ry)], [0, 1, 0], [-sin(ry), 0, cos(ry)]])
        Rz = np.array([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]])
        # R = Rz * Ry * Rx
        R = np.dot(np.dot(Rz, Ry), Rx)
        if Tr_0_inv is None:
            Tr_0_inv = np.linalg.inv(np.vstack((np.hstack((R,np.expand_dims(np.array(t), axis=-1))),
                                                np.array([0, 0, 0, 1]))))
        pose.append(np.dot(Tr_0_inv, np.vstack((np.hstack((R,np.expand_dims(np.array(t), axis=-1))),
                                                np.array([0, 0, 0, 1])))))
    return pose


def proj_latlon(lat, lon):
    myProj = pyproj.Proj(proj='utm', ellps='WGS84')
    x, y = myProj(lon, lat)
    return x, y


class Tracklet:
    def __init__(self, coords, instance, label_file, frame_interval, drive):
        self.coords = coords
        self.points = np.array([(p[0], p[1]) for p in self.coords])
        self.label_file = label_file
        self.drive_date = label_file.split('/')[-3]
        self.drive_name = label_file.split('/')[-2]
        self.frame_interval = frame_interval
        self.drive = drive
        self.instance = instance

    def get_track_coords_xyz(self):
        all_points = []
        for i in range(len(self.coords)):
            cur_point = np.dot(np.append(self.coords[i], 1), self.drive.pose[self.frame_interval[0] + i].T)
            all_points.append(cur_point)
        return np.array(all_points)

    def show_coords_with_pose(self, ax=None, plot_3d=True):
        if ax is None:
            fig = plt.figure()
            if plot_3d:
                ax = fig.gca(projection='3d')
            else:
                ax = fig.gca()
        all_points = self.get_track_coords_xyz()
        if plot_3d is True:
            ax.plot(np.array(all_points)[:, 0], np.array(all_points)[:, 1], np.array(all_points)[:, 2])
        else:
            ax.plot(np.array(all_points)[:, 0], np.array(all_points)[:, 1])
        return ax


class Drive:
    def __init__(self, oxt_file):
        self.oxt_file = oxt_file
        self.oxt_data_files = sorted(glob(oxt_file))
        self.oxt_data = []
        for oxt in self.oxt_data_files:
            f = open(oxt, 'r')
            self.oxt_data.append([float(x) for x in f.read().split(' ')])
        self.pose = convert_oxts_to_pose(self.oxt_data)
        self.lat = [x[0] for x in self.oxt_data]
        self.long = [x[1] for x in self.oxt_data]
        self.altitude = [x[2] for x in self.oxt_data]
        self.roll = [x[3] for x in self.oxt_data]
        self.pitch = [x[4] for x in self.oxt_data]
        self.yaw = [x[5] for x in self.oxt_data]

        self.world_coords = np.array([proj_latlon(p[0], p[1]) for p in zip(self.lat, self.long)])
        self.world_coords -= self.world_coords[0, :]

    def show_pose(self, plot_3d=True):
        l = 3
        A = np.array([[0, 0, 0, 1], [l, 0, 0, 1], [0, 0, 0, 1], [0, l, 0, 1], [0, 0, 0, 1], [0, 0, l, 1]]).T

        fig = plt.figure()
        if plot_3d:
            ax = fig.gca(projection='3d')
        else:
            ax = fig.gca()
        ax.axis('equal')
        all_B = []
        for p in self.pose:
            B = np.dot(p, A)
            all_B.append(B)
            if plot_3d:
                ax.plot3D(B[0, :2], B[1, :2], B[2, :2], 'red')
                ax.plot3D(B[0, 2:4], B[1, 2:4], B[2, 2:4], 'green')
                ax.plot3D(B[0, 4:6], B[1, 4:6], B[2, 4:6], 'blue')
            else:
                ax.plot(B[0, :2], B[1, :2], 'red')
                ax.plot(B[0, 2:4], B[1, 2:4], 'green')
                ax.plot(B[0, 4:6], B[1, 4:6], 'blue')

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        if plot_3d:
            coord_max = np.array(all_B).max()
            coord_min = np.array(all_B).min()
            ax.set_xlim(coord_min, coord_max)
            ax.set_ylim(coord_min, coord_max)
            ax.set_zlim(coord_min, coord_max)
            ax.set_zlabel('Z axis')
        # ax.axis('equal')
        plt.title(self.oxt_file.split('/')[7])
        # plt.show()
        return ax


def get_drive_tracks(track_list, drive):
    return [t for t in track_list if t.drive == drive]


def get_desire_track_files(tracklet_path,train):
    ''' Get videos only from the splits defined in DESIRE: https://arxiv.org/abs/1704.04394
    Splits obtained by the authors:
    all: [1, 2, 5, 9, 11, 13, 14, 15, 17, 18, 27, 28, 29, 32, 48, 51, 52, 56, 57, 59, 60, 70, 84, 91]
    train: [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
    test: [1, 2, 15, 18, 29, 32, 52, 70]
    '''
    #desire_ids = [1, 2, 5, 9, 11, 13, 14, 15, 17, 18, 27, 28, 29, 32, 48, 51, 52, 56, 57, 59, 60, 70, 84, 91]
    if train:
        desire_ids = [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
    else:
        desire_ids = [1, 2, 15, 18, 29, 32, 52, 70]

    tracklet_files = [tracklet_path + '/2011_09_26/2011_09_26_drive_' + str(x).zfill(4) + '_sync/tracklet_labels.xml'
                      for x in desire_ids]
    return tracklet_files


def get_all_3d_tracks(train, draw_tracks=False, plot_3d=False):

    tracklet_path = './kitti_raw_data'
    # tracklet_files = glob(tracklet_path + '*/*/tracklet_labels.xml') # use this to use all files
    tracklet_files = get_desire_track_files(tracklet_path,train)

    drive_list = []
    track_list = []
    all_3d_tracks = []

    for tracklet_file in tracklet_files:
        parsed = XMLParser.parse(tracklet_file)
        track_keys = parsed[0].keys()

        oxt_path = tracklet_file.replace('tracklet_labels.xml', 'oxts/data/*.txt')
        cur_drive = Drive(oxt_path)
        print(cur_drive)
        drive_list.append(cur_drive)

        for track_key in track_keys:
            print(track_key)
            coords = parsed[0][track_key]
            frame_interval = parsed[1][track_key]
            tracklet = Tracklet(coords, track_key, tracklet_file, frame_interval, cur_drive)
            track_list.append(tracklet)
            all_3d_tracks.append(tracklet.get_track_coords_xyz())

        if draw_tracks:
            '''
            Set to True to plot the tracks
            '''
            ax = cur_drive.show_pose(plot_3d=plot_3d)
            drive_tracks = get_drive_tracks(track_list, cur_drive)
            for t in drive_tracks:
                t.show_coords_with_pose(ax, plot_3d=plot_3d)
            # ax.view_init(90, 0)
            #plt.show()
    return all_3d_tracks, drive_list, track_list


if __name__ == "__main__":
    all_3d_tracks, drive_list, track_list = get_all_3d_tracks(draw_tracks=True, plot_3d=False)


