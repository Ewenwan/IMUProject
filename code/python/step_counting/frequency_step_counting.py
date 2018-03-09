import math
import numpy as np
import quaternion
import pandas
from scipy import interpolate

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from algorithms import geometry
from algorithms import icp
from utility import write_trajectory_to_ply


def get_coe(mv, cv, rv, iv):
    """
    This helper function returns the coefficients for
    a*cos^2 + b*sin^2 + c*cos*sin
    """
    c1, c2, c3, c4 = np.dot(rv, mv), np.dot(rv, cv), np.dot(iv, mv), np.dot(iv, cv)
    return np.array([c1 * c1 + c3 * c3, c2 * c2 + c4 * c4, 2 * (c1 * c2 + c3 * c4)])


def get_theta_closed_form(linacce_fft, gyro_fft, target_frq, mv, gv):
    # Compute some intermediate vectors
    cv = np.cross(gv, mv)
    param = np.zeros(3)
    # Linear acceleration
    rv_full, iv_full = np.real(linacce_fft[target_frq]), np.imag(linacce_fft[target_frq])
    rv_half, iv_half = np.real(linacce_fft[target_frq // 2]), np.imag(linacce_fft[target_frq // 2])
    param += get_coe(mv, cv, rv_full, iv_full)  # forward
    param += get_coe(cv, -mv, rv_half, iv_half)  # lateral
    # Gyroscope
    rv_full, iv_full = np.real(gyro_fft[target_frq]), np.imag(gyro_fft[target_frq])
    rv_half, iv_half = np.real(gyro_fft[target_frq // 2]), np.imag(gyro_fft[target_frq // 2])
    param += get_coe(mv, cv, rv_half, iv_half)  # forward
    param += get_coe(cv, -mv, rv_full, iv_full)  # lateral
    return math.pi / 4 - math.atan2(param[0] - param[1], param[2]) / 2


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--window_size', type=int, default=400)
    parser.add_argument('--stride', type=float, default=0.64)
    parser.add_argument('--start_length', type=int, default=5)
    parser.add_argument('--no_ground_truth', action='store_true')

    args = parser.parse_args()

    data_all = pandas.read_csv(args.path + '/processed/data.csv')
    steps = np.genfromtxt(args.path + '/step.txt')

    ts = data_all['time'].values
    orientations = data_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values
    positions = data_all[['pos_x', 'pos_y', 'pos_z']].values
    gravity = data_all[['grav_x', 'grav_y', 'grav_x']].values
    linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values
    gyro = data_all[['gyro_x', 'gyro_y', 'gyro_z']].values
    magnet = data_all[['magnet_x', 'magnet_y', 'magnet_z']].values

    gravity = gravity / np.linalg.norm(gravity, axis=1)[:, None]
    magnet = magnet / np.linalg.norm(magnet, axis=1)[:, None]

    steps = np.concatenate([[[ts[0] + 1, 0]], steps], axis=0)

    step_index = np.zeros(steps.shape[0], dtype=np.int)
    for i in range(step_index.shape[0]):
        step_index[i] = np.argmin(steps[i, 0] >= ts)

    # Compute the magnet vector and gravity vector at each step
    gravity_func = interpolate.interp1d(ts, gravity, axis=0)
    gravity_at_steps = gravity_func(steps[:, 0])

    # Fake accurate magnetometer data from the device orientation, assuming that the device is facing
    # north at the first frame.
    global_north = np.array([0.0, 1.0, 0.0])
    magnet_faked = np.empty(magnet.shape)
    for i in range(magnet.shape[0]):
        rotor = quaternion.as_rotation_matrix(quaternion.quaternion(*orientations[i]))
        magnet_faked[i] = np.matmul(rotor.T, global_north)
    magnet_faked_func = interpolate.interp1d(ts, magnet_faked, axis=0)
    magnet_faked_at_steps = magnet_faked_func(steps[:, 0])

    # Compute ground-truth theta
    positions_at_steps = interpolate.interp1d(ts, positions, axis=0)(steps[:, 0])

    # We also computing the ground truth walking direction to help resolving 180 ambiguity.
    position_dir = positions_at_steps[1:] - positions_at_steps[:-1]
    position_dir[:, 2] = 0
    position_dir = position_dir / np.linalg.norm(position_dir, axis=1)[:, None]
    gt_theta = np.arccos(np.array([np.dot(global_north, pd) for pd in position_dir]))
    gt_theta[position_dir[:, 0] > 0] *= -1
    gt_theta -= gt_theta[0]

    # First compute the target frequency from the vertical acceleration. The vertical acceleration comes from the dot
    # product between the linear acceleration and gravity direction.
    frq = np.zeros(steps.shape[0], dtype=np.int)
    acce_grav = np.array([np.dot(linacce[i], gravity[i]) for i in range(ts.shape[0])])
    for i in range(frq.shape[0]):
        if step_index[i] < args.window_size:
            continue
        acce_segment = acce_grav[step_index[i] - args.window_size:step_index[i]]
        max_frq_loc = np.argmax(np.abs(np.fft.rfft(acce_segment)))
        frq[i] = max_frq_loc

    reference_yaw = magnet_faked_at_steps

    theta_at_step = np.zeros(steps.shape[0])
    test_cf = np.zeros(steps.shape[0])
    delta_phase = np.zeros(steps.shape[0])

    # Compute the yaw angle at each step
    for i in range(0, theta_at_step.shape[0]):
        if step_index[i] < args.window_size:
            continue
        mv, gv = reference_yaw[i], gravity_at_steps[i]
        # Tile compensation for mv
        mv = np.cross(gv, np.cross(mv, gv))
        mv /= np.linalg.norm(mv)

        # First compute the fft of linear acceleration and angular rate
        linacce_segment = linacce[step_index[i] - args.window_size:step_index[i]]
        linacce_fft = np.fft.rfft(linacce_segment, axis=0)
        gyro_segment = gyro[step_index[i] - args.window_size:step_index[i]]
        gyro_fft = np.fft.rfft(gyro_segment, axis=0)

        theta_cf = get_theta_closed_form(linacce_fft, gyro_fft, frq[i], mv, gv)
        theta_at_step[i] = theta_cf

        # Compute the phase difference between the forward acceleration and vertical acceleration.
        forward_dir = mv * math.cos(theta_cf) + np.cross(gv, mv) * math.sin(theta_cf)
        phase_vert = np.angle(np.dot(linacce_fft[frq[i]], gv))
        phase_forward = np.angle(np.dot(linacce_fft[frq[i]], forward_dir))
        delta_phase[i] = math.fabs(phase_vert - phase_forward) * 180 / math.pi

    theta_at_step2 = theta_at_step - theta_at_step[1]
    theta_at_step2[0] = 0

    gt_yaw = np.empty(ts.shape)
    for i in range(gt_yaw.shape[0]):
        gt_yaw[i] = quaternion.as_euler_angles(quaternion.quaternion(*orientations[i]))[0]
    gt_yaw -= gt_yaw[0]
    gt_yaw_at_steps = interpolate.interp1d(ts, gt_yaw, axis=0)(steps[:, 0])

    # Resolve 180 ambiguity
    thres = 1.0
    for i in range(0, theta_at_step2.shape[0] - 1):
        if abs(math.cos(theta_at_step2[i]) - math.cos(gt_theta[i])) > thres or abs(
                        math.sin(theta_at_step2[i]) - math.sin(gt_theta[i])) > thres:
            theta_at_step2[i] += math.pi

    # Dead-reckoning
    track_length = np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=1))
    stride = args.stride
    positions_sc = np.zeros([steps.shape[0], 2])
    for i in range(1, positions_sc.shape[0]):
        positions_sc[i] = positions_sc[i - 1] + stride * np.array([math.cos(theta_at_step2[i - 1]),
                                                                   math.sin(theta_at_step2[i - 1])])

    _, rotation_2d, translation_2d = icp.fit_transformation(positions_sc[:args.start_length, :2],
                                                            positions_at_steps[:args.start_length, :2])
    positions_sc[:, :2] = np.dot(rotation_2d, (positions_sc[:, :2]
                                               - positions_at_steps[0, :2]).T).T + positions_at_steps[0, :2]

    # Interpolate positions at steps back to IMU time stamp and write ply file
    positions_sc = np.concatenate([[positions_sc[0]], positions_sc, [positions_sc[-1]]], axis=0)
    step_ext = np.concatenate([[ts[0] - 1], steps[:, 0], [ts[-1] + 1]], axis=0)
    positions_at_imu = interpolate.interp1d(step_ext, positions_sc, axis=0)(ts)
    positions_at_imu = np.concatenate([positions_at_imu, np.zeros([ts.shape[0], 1])], axis=1)

    # Write the result
    print('Writing to csv')
    data_mat = np.zeros([ts.shape[0], 10], dtype=float)
    column_list = ['time', 'pos_x', 'pos_y', 'pos_z', 'speed_x', 'speed_y', 'speed_z', 'bias_x', 'bias_y', 'bias_z']
    data_mat[:, 0] = ts
    data_mat[:, 1:4] = positions_at_imu

    out_dir = args.path + '/result_frq_step/'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    data_pandas = pandas.DataFrame(data_mat, columns=column_list)
    data_pandas.to_csv(out_dir + '/result_frq_step.csv')

    write_trajectory_to_ply.write_ply_to_file(out_dir + '/result_trajectory_frq_step.ply', positions_at_imu,
                                              orientations, trajectory_color=(80, 80, 80), length=0,
                                              interval=300, num_axis=0)

    print('All done')