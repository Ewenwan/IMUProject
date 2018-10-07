import numpy as np
import pandas
import quaternion
from scipy.ndimage.filters import gaussian_filter1d
import math
from os import path as osp
import sys
sys.path.append(osp.join(osp.abspath(__file__), '..'))

from algorithms import geometry

_nano_to_sec = 1e09


def compute_yaw_in_tango(orientations):
    yaw = np.empty(orientations.shape[0])
    for i in range(orientations.shape[0]):
        _, _, yaw[i] = geometry.quaternion_to_euler(*orientations[i])
    return yaw


def compute_north_in_tango(orientations, magnet, inlier_threshold=0.1):
    north_in_tango = np.empty(magnet.shape[0])
    for i in range(magnet.shape[0]):
        q = quaternion.quaternion(*orientations[i])
        mg = (q * quaternion.quaternion(0, *magnet[i]) * q.conj()).vec[0:2]
        north_in_tango[i] = math.atan2(mg[1], mg[0])
    median_north = np.median(north_in_tango)
    diff = np.abs(north_in_tango - median_north)
    inlier = north_in_tango[diff < inlier_threshold]
    return np.mean(inlier)


def compute_mfn_target_angle(imu_all):
    orientations = imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    magnet = imu_all[['magnet_x', 'magnet_y', 'magnet_z']].values
    yaw_in_tango = compute_yaw_in_tango(orientations)
    north_in_tango = compute_north_in_tango(orientations, magnet)
    yaw_in_ned = np.mod(yaw_in_tango - north_in_tango, 2 * math.pi)[:, np.newaxis]
    return yaw_in_ned


def compute_mfn_target_sincos(imu_all):
    yaw_in_ned = compute_mfn_target_angle(imu_all)
    return np.concatenate([np.sin(yaw_in_ned), np.cos(yaw_in_ned)], axis=1)


def load_datalist(root_dir, data_list, feature_column, target_type, *args, **kwargs):
    train_features = []
    train_targets = []

    for data in data_list:
        print('Loading ', data)
        imu_all = pandas.read_csv(osp.join(root_dir, data, 'processed/data.csv'))

        feature = imu_all[feature_column]
        if 'feature_sigma' in kwargs:
            feature = gaussian_filter1d(imu_all[feature_column].values, sigma=kwargs['feature_sigma'], axis=0)
        if target_type == 'angle':
            target = compute_mfn_target_angle(imu_all)
        elif target_type == 'sc':
            target = compute_mfn_target_sincos(imu_all)
        elif target_type == 'angle_cls':
            target = compute_mfn_target_angle(imu_all)
            angle_step = kwargs['angle_step'] if 'angle_step' in kwargs else 2.0
            target = (target / angle_step).astype(np.int)
        else:
            raise ValueError('target_type must be one of "angle" or "sc"')
        if 'target_sigma' in kwargs:
            target = gaussian_filter1d(target, sigma=kwargs['target_sigma'], axis=0)
        train_features.append(feature)
        train_targets.append(target)

    return train_features, train_targets
