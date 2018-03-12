import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas
import argparse
import quaternion
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.metrics import mean_squared_error

sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')

from speed_regression import training_data as td
from algorithms import geometry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--id', default='full', type=str)

    filter_sigma = 20.0

    args = parser.parse_args()
    result_path = args.dir + '/result_' + args.id
    result = pandas.read_csv(result_path + '/result_'+args.id+'.csv')
    data_all = pandas.read_csv(args.dir + '/processed/data.csv')

    # Be careful about invalid regression value belonging to "transition" model.
    kMaxSpeed = 100
    regression_result = np.genfromtxt(result_path + '/regression_'+args.id+'.txt')
    regressed_magnitude = np.linalg.norm(regression_result[:, 1:], axis=1)
    regression_valid_index = regressed_magnitude < 100

    constraint_ind = regression_result[regression_valid_index, 0].astype(int)
    local_speed = regression_result[regression_valid_index, 1:]

    ts = result['time'].values
    ts -= ts[0]

    pos_res = result[['pos_x', 'pos_y', 'pos_z']].values

    position_gt = data_all[['pos_x', 'pos_y', 'pos_z']].values
    orientation_gt = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values

    traj_length = sum(np.linalg.norm(position_gt[1:] - position_gt[:-1], axis=1))
    mean_offset = np.average(np.linalg.norm(position_gt - pos_res, axis=1))
    print('Trajectory length: {:.3f}m, {:.3f}s, mean offset: {:.3f}m ({:.3f})'.format(traj_length, ts[-1] - ts[0],
          mean_offset, mean_offset / traj_length))

    ls_gt = td.compute_local_speed_with_gravity(ts, position_gt, orientation_gt, gravity)
    ls_gt = gaussian_filter1d(ls_gt, sigma=30.0, axis=0)
    ls_gt = ls_gt[constraint_ind]

    print('Regression error: {:.3f}, {:.3f}'.format(
        mean_squared_error(ls_gt[:, 0], local_speed[:, 0]),
        mean_squared_error(ls_gt[:, 2], local_speed[:, 2])))

    font = {'family': 'serif', 'size': 40}
    plt.rc('font', **font)

    axes_glob = [0, 1]
    axes_local = [0, 2]

    lines_imu = []
    lines_tango = []

    ylabels = ['X Speed (m/s)', 'Z Speed (m/s)']
    fig_ls = plt.figure('Local speed', figsize=(24, 18))
    linewidth = 1.8

    for i in range(0, 2):
        plt.subplot(211+i)
        if i == 0:
            plt.xlabel('Time(s)')
        plt.ylabel(ylabels[i])
        plt.locator_params(nbins=5, axis='y')
        lines_imu += plt.plot(ts[constraint_ind], local_speed[:, axes_local[i]], 'b', lw=linewidth)
        lines_tango += plt.plot(ts[constraint_ind], ls_gt[:, axes_local[i]], 'r', lw=linewidth)
    # plt.figlegend([lines_imu[-1], lines_tango[-1]], ['Our method', 'Tango (Ground truth)'],
    #               loc='upper center', ncol=2, labelspacing=0.)
    fig_ls.savefig(result_path + '/regression.png', bbox_inches='tight')
    # fig_ls.show()
