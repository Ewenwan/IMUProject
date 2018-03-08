import numpy as np
import pandas
import argparse
from scipy import interpolate
import matplotlib.pyplot as plt
import quaternion

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from algorithms import icp
from algorithms import geometry
from utility import write_trajectory_to_ply

nano_to_sec = 1e09

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--start_portion_length', default=1000, type=int)

    print('Loading...')
    args = parser.parse_args()
    step_data = np.genfromtxt(args.dir + '/step.txt')
    step_data[:, 0] /= nano_to_sec

    data_all = pandas.read_csv(args.dir + '/processed/data.csv')

    ts = data_all['time'].values / nano_to_sec
    positions = data_all[['pos_x', 'pos_y', 'pos_z']].values
    is_gt_valid = np.linalg.norm(np.sum(positions, axis=0)) > 1e-05
    # orientations = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    orientations = data_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values
    gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
    track_length = np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=1))
    # step_stride = track_length / step_data[-1][1]
    step_stride = 0.67
    if is_gt_valid:
        print('Track length: {:.3f}m, stride length: {:.3f}m'.format(track_length, step_stride))

    # Interpolate step counts to Tango's sample rate.
    step_data = np.concatenate([np.array([[ts[0] - 1, 0]]), step_data, np.array([[ts[-1] + 1, step_data[-1][1]]])],
                               axis=0)
    step_func = interpolate.interp1d(step_data[:, 0], step_data[:, 1])
    step_interpolated = step_func(ts)

    # Compute positions by dead-reckoning
    step_length = step_stride * (step_interpolated[1:] - step_interpolated[:-1])
    position_from_step = np.zeros(positions.shape)
    if is_gt_valid:
        position_from_step[0] = positions[0]

    # Compute the forward speed. Notice that the device is not necessarily facing horizontally. We need to take the
    # gravity direction into consideration.
    local_gravity_dir = np.array([0., 1., 0.])
    forward_dir = []
    rotation_horizontal = []
    for i in range(gravity.shape[0]):
        rotor = geometry.quaternion_from_two_vectors(local_gravity_dir, gravity[i])
        # rotor = geometry.quaternion_from_two_vectors(gravity[i], local_gravity_dir)
        # forward_dir.append(rotor * quaternion.quaternion(0., 0., 0., -1.) * rotor.conj())
        forward_dir.append(rotor * quaternion.quaternion(0, 0, 0, -1) * rotor.conj())
        rotation_horizontal.append(quaternion.quaternion(*orientations[i]))


    for i in range(1, positions.shape[0]):
        q = rotation_horizontal[i - 1]
        segment = (q * forward_dir[i] * q.conj()).vec * step_length[i - 1]
        position_from_step[i] = position_from_step[i-1] + segment

    # If the ground truth is presented, ind a transformation to align the start portion of estimated
    # track and the ground truth track.
    if is_gt_valid:
        start_length = args.start_portion_length
        _, rotation_to_gt, translation_to_gt = icp.fit_transformation(position_from_step, positions)
        position_from_step = np.dot(rotation_to_gt, (position_from_step - positions[0]).T).T + positions[0]

        _, rotation_2d, translation_2d = icp.fit_transformation(position_from_step[:start_length, :2],
                                                            positions[:start_length, :2])
        position_from_step[:, :2] = np.dot(rotation_2d, (position_from_step[:, :2]
                                                     - positions[0, :2]).T).T + positions[0, :2]

    # rotation_combined = np.identity(3)
    # rotation_combined[:2, :2] = rotation_2d
    # rotatioin_combined = rotation_combined * rotation_to_gt
    # output_orientations = orientations[:]
    # for i in range(orientations.shape[0]):
    #     rotor = quaternion.from_rotation_matrix(rotation_combined)
    #     out_orientation = rotor * quaternion.quaternion(*orientations[i]) * rotor.conj()
    #     output_orientations[i] = quaternion.as_float_array(out_orientation)

    print('Writing to csv')
    data_mat = np.zeros([ts.shape[0], 10], dtype=float)
    column_list = ['time', 'pos_x', 'pos_y', 'pos_z', 'speed_x', 'speed_y', 'speed_z', 'bias_x', 'bias_y', 'bias_z']
    data_mat[:, 0] = ts
    data_mat[:, 1:4] = position_from_step

    out_dir = args.dir + '/result_step/'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    data_pandas = pandas.DataFrame(data_mat, columns=column_list)
    data_pandas.to_csv(out_dir + '/result_step.csv')

    write_trajectory_to_ply.write_ply_to_file(out_dir + '/result_trajectory_step.ply', position_from_step,
                                              orientations, trajectory_color=(80, 80, 80), length=0,
                                              interval=300, num_axis=0)

    print('All done')
