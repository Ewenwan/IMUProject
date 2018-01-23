import os
import numpy as np
import quaternion
import argparse
import pandas
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

import sys
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')

from algorithms import geometry


def compute_gravity_speed(ts, linacce, orientation, gravity):
    directed = np.empty(linacce.shape, dtype=float)
    for i in range(directed.shape[0]):
        q = quaternion.quaternion(*orientation[i])
        directed[i] = (q * quaternion.quaternion(1.0, *linacce[i]) * q.conj()).vec
    speed = np.cumsum(directed[:-1] * (ts[1:, None] - ts[:-1, None]), axis=0)
    speed = np.concatenate([np.array([[0., 0., 0.]]), speed], axis=0)
    local_grav_dir = np.array([0., 1., 0.])
    for i in range(speed.shape[0]):
        q = quaternion.quaternion(*orientation[i])
        rotor = geometry.quaternion_from_two_vectors(gravity[i], local_grav_dir)
        ls = q.conj() * quaternion.quaternion(1.0, *speed[i]) * q
        speed[i] = (rotor * ls * rotor.conj()).vec
    return speed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--id', type=str, default='full')
    args = parser.parse_args()

    data_all = pandas.read_csv(args.dir + '/processed/data.csv')

    ts = data_all['time'].values / 1e09
    ts -= ts[0]
    ts /= 3
    filter_sigma = 30.0

    bias_interval = 150
    bias_ind = np.arange(bias_interval, ts.shape[0], bias_interval)
    bias_start_id = 15
    bias_end_id = 21

    linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values
    gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

    speed_raw = compute_gravity_speed(ts, linacce, orientation, gravity)

    result_all = pandas.read_csv(args.dir + '/result_{0}/result_{0}.csv'.format(args.id))
    regression_result = np.genfromtxt(args.dir + '/result_{0}/regression_{0}.txt'.format(args.id))
    regression_result = regression_result[0:-1:3]
    constraint_ind = regression_result[:, 0].astype(int)
    # constraint = gaussian_filter1d(regression_result[:, 1:], sigma=5, axis=0)
    constraint = regression_result[:, 1:]
    bias = result_all[['bias_x', 'bias_y', 'bias_z']].values
    corrected_linacce = gaussian_filter1d(linacce + bias, sigma=filter_sigma, axis=0)
    linacce = gaussian_filter1d(linacce, sigma=filter_sigma, axis=0)

    speed_corrected = compute_gravity_speed(ts, corrected_linacce, orientation, gravity)

    # font_config = {'family': 'serif',
    #                'size': 120}
    # linew = 18
    # markersize = 100
    #
    # plt.rc('font', **font_config)
    # fig = plt.figure('Sparse grid', figsize=(130, 25))

    font_config = {'family': 'serif',
                   'size': 18}
    linew = 5
    markersize = 15

    plt.rc('font', **font_config)
    fig = plt.figure('Sparse grid', figsize=(16, 8))

    glob_start_id = bias_ind[bias_start_id]
    glob_end_id = bias_ind[bias_end_id]

    bias_end_id = min(bias_ind.shape[0] - 1, bias_end_id + 1)

    # First draw the plot for low-frequency bias.
    # for i in range(1):
    #     plt.subplot(111 + i * 2, axis_bgcolor='black')
    #     plt.ylabel('Acceleration (m/s2)')
    #     plt.xlabel('Time (s)')
    #     x_min = min(ts[bias_ind[bias_start_id]], ts[glob_start_id]) - 0.1
    #     x_max = max(ts[bias_ind[bias_end_id - 1]], ts[glob_end_id]) + 0.1
    #     plt.xlim(x_min, x_max)
    #     plt.plot(ts[glob_start_id:glob_end_id], linacce[glob_start_id:glob_end_id, i], color=(0, 0.7, 0), lw=linew)
    #     plt.plot(ts[glob_start_id:glob_end_id], corrected_linacce[glob_start_id:glob_end_id, i], color='b', lw=linew)
    #     plt.plot(ts[bias_ind[bias_start_id:bias_end_id]], bias[bias_ind[bias_start_id:bias_end_id], i],
    #              color=(1, 1, .0), lw=linew, marker='o', markersize=markersize)
    # plt.legend(['Sparse bias', 'Raw acceleration', 'Corrected acceleration'], loc='lower left')

    # Then draw the plot for velocity constraints.
    con_start_id = max(np.where(constraint_ind > glob_start_id)[0][0] - 1, 0)
    con_end_id = min(np.where(constraint_ind > glob_end_id)[0][0] + 1, constraint_ind.shape[0] - 1)
    print('con_start_id: %d, con_end_id: %d' % (con_start_id, con_end_id))

    glob_start_id = constraint_ind[con_start_id]
    glob_end_id = constraint_ind[con_end_id]

    for i in [0]:
        plt.subplot(111 + i * 2, axis_bgcolor='black')
        plt.ylabel('Velocity (m/s)')
        plt.xlabel('Time (s)')
        # x_min = min(ts[constraint_ind[con_start_id]], ts[glob_start_id]) - 0.1
        # x_max = max(ts[constraint_ind[con_end_id - 1]], ts[glob_end_id]) + 0.1
        x_min, x_max = 3.9, 5.8
        plt.xlim(x_min, x_max)
        plt.plot(ts[glob_start_id:glob_end_id], speed_raw[glob_start_id:glob_end_id, i], color=(0, 0.7, 0), lw=linew)
        plt.plot(ts[glob_start_id:glob_end_id], speed_corrected[glob_start_id:glob_end_id, i] - 0.05, color='b', lw=linew)
        # plt.plot(ts[constraint_ind[con_start_id:con_end_id]], constraint[con_start_id:con_end_id, i], color=(.8, .5, 0),
        #          lw=linew, marker='^', markersize=markersize)

    # plt.subplot(324)
    # plt.ylabel('Velocity (m/s)')
    # plt.xlabel('Time (s)')
    # plt.plot(ts[glob_start_id:glob_end_id], speed_raw[glob_start_id:glob_end_id, 1], color='r', lw=linew)
    # plt.plot(ts[glob_start_id:glob_end_id], speed_corrected[glob_start_id:glob_end_id, 1], color='b', lw=linew)
    # plt.plot(ts[constraint_ind[con_start_id:con_end_id]], np.zeros(con_end_id - con_start_id), color=(.8, .5, 0),
    #          lw=linew, marker='^', markersize=markersize)

    # plt.legend(['Predicted speed', 'Before correction', 'After correction'], loc='lower left')

    # fig.savefig(args.dir + '/result_{}/alg1.png'.format(args.id), bbox_inches='tight')
    plt.show()
