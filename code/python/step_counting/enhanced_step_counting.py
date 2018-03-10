import os
import sys
import math
import numpy as np
import quaternion
import pandas
from scipy.ndimage.filters import maximum_filter1d
from scipy.signal import lfilter, kaiserord, firwin
from scipy import interpolate

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from algorithms import geometry
from algorithms import icp
from utility import write_trajectory_to_ply


def detect_steps(linacce, gravity, labels, threshold=1.0, min_step_interval=10, fpr_interval=24):
    acce_grav = geometry.align_3dvector_with_gravity(linacce, gravity)
    channels = np.stack([acce_grav[:, 1], linacce[:, 1], linacce[:, 1], linacce[:, 1]], axis=1)
    chn = np.array([channels[i, labels[i]] for i in range(channels.shape[0])])
    out_steps = [-1]
    buffered_peak_loc = -1
    for i in range(1, chn.shape[0] - 1):
        if chn[i] >= chn[i-1] and chn[i] >= chn[i+1] and chn[i] > threshold:
            if i - out_steps[-1] < min_step_interval:
                buffered_peak_loc = -1
                continue
            buffered_peak_loc = i
        else:
            if buffered_peak_loc > 0 and i - buffered_peak_loc >= fpr_interval:
                out_steps.append(buffered_peak_loc)
                buffered_peak_loc = -1
    out_steps = np.array(out_steps[1:])
    return out_steps, chn


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--start_length', type=int, default=2500)
    parser.add_argument('--stride', type=float, default=-1.0)
    parser.add_argument('--placement', type=str, default='handheld')
    parser.add_argument('--height', type=float, default=1.7)
    parser.add_argument('--k', type=float, default=0.3139)

    args = parser.parse_args()

    class_map = {'bag': 2, 'handheld': 0, 'leg': 1, 'body': 3}
    data_all = pandas.read_csv(args.path + '/processed/data.csv')

    ts = data_all[['time']].values
    linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values
    orientations = data_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values
    positions = data_all[['pos_x', 'pos_y', 'pos_z']].values
    positions[:, 2] = 0
    gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
    steps = np.genfromtxt(args.path + '/step.txt')

    labels = np.ones(ts.shape[0], dtype=np.int) * class_map[args.placement]

    # Pro-processing
    sample_rate = 200
    nyq_rate = sample_rate / 2.0
    transition_width = 5.0 / nyq_rate
    ripple_db = 60.0
    order, beta = kaiserord(ripple_db, transition_width)
    cutoff_hz = 5
    taps = firwin(order, cutoff_hz / nyq_rate, window=('kaiser', beta))
    threshold = 1.0

    linacce_filtered = lfilter(taps, 1.0, linacce, axis=0)
    acce_grav = geometry.align_3dvector_with_gravity(linacce_filtered, gravity)

    print('Detecting steps')
    step_locs, chn = detect_steps(linacce_filtered, gravity, labels, threshold)
    print('Step from system: %d, estimated: %d' % (steps[-1, 1], step_locs.shape[0]))

    # Compose the detected steps
    est_steps = np.concatenate([ts[step_locs], np.arange(step_locs.shape[0])[:, np.newaxis]], axis=1)
    est_steps = np.concatenate([np.array([[ts[0, 0] + 1, 0.0]]), est_steps], axis=0)
    step_locs = np.concatenate([[1], step_locs], axis=0)

    nano_to_sec = 1e09
    step_frq = nano_to_sec / (est_steps[1:, 0] - est_steps[:-1, 0])
    step_frq = np.concatenate([[0.0], step_frq], axis=0)

    step_stride = np.ones(step_frq.shape[0]) * args.stride
    if args.stride < 0:
        step_stride = args.height * args.k * np.sqrt(step_frq)

    # Compute the yaw
    yaw = np.empty(orientations.shape[0])
    quat_array = quaternion.as_quat_array(orientations)
    local_gravity = np.array([0, 1, 0])
    for i in range(yaw.shape[0]):
        rotor_gravity = geometry.quaternion_from_two_vectors(local_gravity, gravity[i])
        yaw[i] = quaternion.as_euler_angles(quat_array[i])[0]

    yaw[yaw < 0] += 2 * math.pi

    yaw_func = interpolate.interp1d(ts[:, 0], yaw)
    yaw_at_steps = yaw_func(est_steps[:, 0])

    gt_func = interpolate.interp1d(ts[:, 0], positions, axis=0)
    gt_at_steps = gt_func(est_steps[:, 0])

    positions_sc = np.zeros([est_steps.shape[0], 2])
    for i in range(1, positions_sc.shape[0]):
        positions_sc[i] = positions_sc[i - 1] + step_stride[i - 1] * np.array([
            math.cos(yaw_at_steps[i - 1]), math.sin(yaw_at_steps[i - 1])])

    positions_sc = np.concatenate([[positions_sc[0]], positions_sc, [positions_sc[-1]]], axis=0)
    step_ext = np.concatenate([[ts[0, 0] - 1], est_steps[:, 0], [ts[-1, 0] + 1]], axis=0)
    positions_at_imu = interpolate.interp1d(step_ext, positions_sc, axis=0)(ts[:, 0])
    positions_at_imu = np.concatenate([positions_at_imu, np.zeros([ts.shape[0], 1])], axis=1)

    # Register the estimated trajectory to the ground truth
    _, rotation_2d, translation_2d = icp.fit_transformation(positions_at_imu[:args.start_length, :2],
                                                            positions[:args.start_length, :2])
    positions_at_imu[:, :2] = np.dot(rotation_2d, (positions_at_imu[:, :2]
                                                   - positions[0, :2]).T).T + positions[0, :2]

    # Write the result
    print('Writing to csv')
    data_mat = np.zeros([ts.shape[0], 10], dtype=float)
    column_list = ['time', 'pos_x', 'pos_y', 'pos_z', 'speed_x', 'speed_y', 'speed_z', 'bias_x', 'bias_y', 'bias_z']
    data_mat[:, 0] = ts[:, 0]
    data_mat[:, 1:4] = positions_at_imu

    out_dir = args.path + '/result_enh_step/'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    data_pandas = pandas.DataFrame(data_mat, columns=column_list)
    data_pandas.to_csv(out_dir + '/result_enh_step.csv')

    write_trajectory_to_ply.write_ply_to_file(out_dir + '/result_trajectory_enh_step.ply', positions_at_imu,
                                              orientations, trajectory_color=(80, 80, 80), length=0,
                                              interval=300, num_axis=0)

    print('All done')