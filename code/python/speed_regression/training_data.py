import os
import sys
import math
import argparse
import numpy as np
import quaternion
import pandas
from scipy.fftpack import fft
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt

sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')
from algorithms import geometry


class TrainingDataOption:
    def __init__(self, sample_step=10, window_size=200, feature='direct_gravity', target='local_speed_gravity'):
        self.sample_step_ = sample_step
        self.window_size_ = window_size
        self.feature_ = feature
        self.target_ = target
        self.nanoToSec = 1000000000.0


def low_pass_filter(data, alpha):
    output = np.copy(data)
    for i in range(1, output.shape[0]):
        output[i] = (1.0 - alpha) * output[i-1] + alpha * data[i]
    return output


def compute_fourier_features(data, samples, window_size, threshold, discard_direct=False):
    """
    Compute fourier coefficients as feature vector
    :param data: NxM array for N samples with M dimensions
    :return: Nxk array
    """
    skip = 0
    if discard_direct:
        skip = 1
    features = np.empty([samples.shape[0], data.shape[1] * (threshold - skip)], dtype=np.float)
    for i in range(samples.shape[0]):
        features[i, :] = np.abs(fft(data[samples[i]-window_size:samples[i]], axis=0)[skip:threshold]).flatten()
    return features


def compute_direct_features(data, samples_points, window_size, sigma=-1):
    features = np.empty([samples_points.shape[0], data.shape[1] * window_size], dtype=np.float)
    for i in range(samples_points.shape[0]):
        data_slice = data[samples_points[i] - window_size:samples_points[i]]
        if sigma > 0:
            data_slice = gaussian_filter1d(data_slice, sigma, axis=0)
        features[i, :] = data_slice.flatten()
    return features


def compute_direct_feature_gravity(gyro, linacce, gravity, samples, window_size, sigma=-1):
    # We scale the gyro data by 0.1, such that it's guaranteed to lie within Pi/2. This prevents some problems when
    # converting between the Euler angle and quaternion representation.\
    # gyro_scaled = gyro / 5.
    # gyro_gravity = geometry.align_eular_rotation_with_gravity(gyro_scaled, gravity)
    # gyro_gravity = gyro_gravity * 5.
    gyro_gravity = gyro
    linacce_gravity = geometry.align_3dvector_with_gravity(linacce, gravity)
    return compute_direct_features(np.concatenate([gyro_gravity, linacce_gravity], axis=1), samples_points=samples,
                                   window_size=window_size, sigma=sigma)


def compute_speed(time_stamp, position, sample_points=None):
    """
    Compute speed vector giving position and time_stamp
    :param time_stamp:
    :param position:
    :param sample_points:
    :return:
    """
    if sample_points is None:
        sample_points = np.arange(0, time_stamp.shape[0], dtype=int)
    sample_points[-1] = min(sample_points[-1], time_stamp.shape[0] - 2)
    speed = (position[sample_points+1] - position[sample_points]) / (time_stamp[sample_points+1] - time_stamp[sample_points])[:, None]
    return speed


def compute_local_speed(time_stamp, position, orientation, sample_points=None):
    """
    Compute the speed in local (IMU) frame
    :param time_stamp:
    :param position: Nx3 array of positions
    :param orientation: Nx4 array of orientations as quaternion
    :param sample_points:
    :return: Nx3 array
    """
    if sample_points is None:
        sample_points = np.arange(0, time_stamp.shape[0], dtype=int)
    sample_points[-1] = min(sample_points[-1], time_stamp.shape[0] - 2)
    speed = compute_speed(time_stamp, position, sample_points)
    for i in range(speed.shape[0]):
        q = quaternion.quaternion(*orientation[sample_points[i]])
        speed[i] = (q.conj() * quaternion.quaternion(1.0, *speed[i]) * q).vec
    return speed


def compute_local_speed_with_gravity(time_stamp, position, orientation, gravity,
                                     sample_points=None, local_gravity=np.array([0., 1., 0.])):
    if sample_points is None:
        sample_points = np.arange(0, time_stamp.shape[0], dtype=int)
    sample_points[-1] = min(sample_points[-1], time_stamp.shape[0] - 2)
    local_speed = compute_local_speed(time_stamp, position, orientation, sample_points)
    # rotate the local speed such at the gravity is along $local_gravity direction
    for i in range(local_speed.shape[0]):
        g = gravity[sample_points[i]]
        rot_q = geometry.quaternion_from_two_vectors(g, local_gravity)
        local_speed[i] = (rot_q * quaternion.quaternion(1.0, *local_speed[i]) * rot_q.conj()).vec
    return local_speed


def compute_acceleration_bias_with_gravity(time_stamp, positions, linacce, gravity,
                                           sample_points=None, local_gravity=np.array([0., 1., 0.])):
    if sample_points is None:
        sample_points = np.arange(0, time_stamp.shape[0], dtype=int)
    sample_points[-1] = min(sample_points[-1], time_stamp.shape[0] - 2)

    dt = time_stamp[1:] - time_stamp[:-1]
    if dt.ndim == 1:
        dt = dt[:, None]

    speed = (positions[1:] - positions[:-1]) / dt
    speed = np.concatenate([np.zeros([1, 3]), speed], axis=0)
    acce = (speed[1:] - speed[:-1]) / dt
    acce = np.concatenate([np.zeros([1, 3]), acce], axis=0)
    # Double differentiation introduces large noises. We pre-filter the linear acceleration with sigma=10
    pre_filter_sigma = 10.0
    acce = gaussian_filter1d(acce, sigma=pre_filter_sigma, axis=0)
    acce_grav = geometry.align_3dvector_with_gravity(acce, gravity, local_gravity)
    linacce_grav = geometry.align_3dvector_with_gravity(linacce, gravity, local_gravity)
    return acce_grav[sample_points] - linacce_grav[sample_points]


def compute_delta_angle(time_stamp, position, orientation, sample_points=None,
                        local_axis=quaternion.quaternion(1.0, 0., 0., -1.)):
    """
    Compute the cosine between the moving direction and viewing direction
    :param time_stamp: Time stamp
    :param position: Position. When passing Nx2 array, compute ignore z direction
    :param orientation: Orientation as quaternion
    :param local_axis: the viewing direction in the device frame. Default is set w.r.t. to android coord frame
    :return:
    """
    if sample_points is None:
        sample_points = np.arange(0, time_stamp.shape[0], dtype=int)
    epsilon = 1e-10
    speed_dir = compute_speed(time_stamp, position)
    speed_dir = np.concatenate([np.zeros([1, position.shape[1]]), speed_dir], axis=0)
    speed_mag = np.linalg.norm(speed_dir, axis=1)
    cos_array = np.zeros(sample_points.shape[0], dtype=float)
    valid_array = np.empty(sample_points.shape[0], dtype=bool)
    for i in range(sample_points.shape[0]):
        if speed_mag[sample_points[i]] <= epsilon:
            valid_array[i] = False
        else:
            q = quaternion.quaternion(*orientation[sample_points[i]])
            camera_axis = (q * local_axis * q.conj()).vec[:position.shape[1]]
            cos_array[i] = min(np.dot(speed_dir[sample_points[i]], camera_axis) / speed_mag[sample_points[i]], 1.0)
            valid_array[i] = True
    return cos_array, valid_array


def get_training_data(data_all, imu_columns, option, sample_points=None, extra_args=None):
    """
    Create training data.
    :param data_all: The whole dataset. Must include 'time' column and all columns inside imu_columns
    :param imu_columns: Columns used for constructing feature vectors. Fields must exist in the dataset
    :param option: An instance of TrainingDataOption
    :param sample_points: an array of locations where the data sample is computed. If not provided, samples are
           uniformly distributed based on option.sample_step_
    :param extra_args: Extra arguments.
    :return: [Nx(d+1)] array. Target value is appended at back
    """
    N = data_all.shape[0]
    if sample_points is None:
        sample_points = np.arange(option.window_size_,
                                  N - 1,
                                  option.sample_step_,
                                  dtype=int)
    assert sample_points[-1] < N
    pose_data = data_all[['pos_x', 'pos_y', 'pos_z']].values
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    data_used = data_all[imu_columns].values
    time_stamp = data_all['time'].values / 1e09
    gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values

    targets = None

    if option.target_ == 'speed_magnitude':
        targets = np.linalg.norm(compute_speed(time_stamp, pose_data), axis=1)
    elif option.target_ == 'angle':
        targets, valid_array = compute_delta_angle(time_stamp, pose_data, orientation)
    elif option.target_ == 'local_speed':
        targets = compute_local_speed(time_stamp, pose_data, orientation)
    elif option.target_ == 'local_speed_gravity':
        targets = compute_local_speed_with_gravity(time_stamp, pose_data, orientation, gravity)
    elif option.target_ == 'acce_bias_gravity':
        linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values
        targets = compute_acceleration_bias_with_gravity(time_stamp, pose_data, linacce, gravity)

    if extra_args is not None:
        if 'target_smooth_sigma' in extra_args:
            targets = gaussian_filter1d(targets, sigma=extra_args['target_smooth_sigma'], axis=0)

    targets = targets[sample_points]

    gaussian_sigma = -1
    if extra_args is not None:
        if 'feature_smooth_sigma' in extra_args:
            gaussian_sigma = extra_args['feature_smooth_sigma']

    if option.feature_ == 'direct':
        features = compute_direct_features(data_used, sample_points, option.window_size_, gaussian_sigma)
    elif option.feature_ == 'fourier':
        print('Additional parameters: ', extra_args)
        features = compute_fourier_features(data_used, sample_points, option.window_size_, extra_args['frq_threshold'],
                                            extra_args['discard_direct'])
    elif option.feature_ == 'direct_gravity':
        features = compute_direct_feature_gravity(data_used[:, :3], data_used[:, -3:],
                                                  gravity, sample_points, option.window_size_, gaussian_sigma)
    else:
        print('Feature type not supported: ' + option.feature_)
        raise ValueError

    return features, targets


def split_data(data, ratio=0.3):
    """
    Randomly split data set
    :param data: all data
    :param ratio: ratio of hold-off set
    :return: set1, set2
    """
    mask = np.random.random(data.shape[0]) < ratio
    return data[mask], data[~mask]


def test_decompose_speed(data_all):
    """
    Unit test: decompose the speed to 'forward' and 'tangent' direction, and integrate back
    :param data_all:
    :return: positions
    """
    nano_to_sec = 1e09
    num_samples = data_all.shape[0]
    time_stamp = data_all['time'].values / nano_to_sec
    position_xy = data_all[['pos_x', 'pos_y']].values
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

    step = 1
    sample_points = np.arange(0, num_samples, step, dtype=int)
    moving_dir = (position_xy[sample_points[1:]] -
                  position_xy[sample_points[:-1]]) / (time_stamp[sample_points[1:]] -
                                                      time_stamp[sample_points[:-1]])[:, None]
    moving_dir = np.concatenate([moving_dir, [moving_dir[-1]]], axis=0)
    moving_mag = np.linalg.norm(moving_dir, axis=1)
    speed_decomposed = np.zeros([sample_points.shape[0], 2], dtype=float)
    camera_axis_local = quaternion.quaternion(1.0, 0.0, 0.0, -1.0)
    for i in range(sample_points.shape[0]):
        if moving_mag[i] < 1e-011:
            continue
        q = quaternion.quaternion(*orientation[sample_points[i]])
        camera_dir = (q * camera_axis_local * q.conj()).vec[:2]
        cos_theta = np.dot(camera_dir, moving_dir[i]) / (moving_mag[i] * np.linalg.norm(camera_dir))
        sin_theta = math.sqrt(1.0 - cos_theta**2)
        rot_mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        moving_dir2 = np.dot(rot_mat, camera_dir) * moving_mag[i]
        if i % 100 == 0:
            print('-----------------')
            print(moving_dir[i])
            print(moving_dir2)
        speed_decomposed[i] = np.dot(rot_mat, camera_dir) * moving_mag[i]

    # Get the position by integrating
    from utility import write_trajectory_to_ply
    position_inte_xy = np.cumsum((speed_decomposed[1:]+speed_decomposed[:-1]) / 2.0
                                 * (time_stamp[sample_points[1:]] - time_stamp[sample_points[:-1]])[:, None], axis=0)
    position_inte_xy = np.insert(position_inte_xy, 0, 0., axis=0)
    position_inte_xy += position_xy[sample_points[0]]
    position_output = np.concatenate([position_inte_xy, np.zeros([sample_points.shape[0], 1])], axis=1)
    write_trajectory_to_ply.write_ply_to_file('test_decompose_{}.ply'.format(step),
                                              position_output, orientation[sample_points])
    return position_output


# for tests
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--window', default=300, type=int)
    parser.add_argument('--step', default=50, type=int)
    parser.add_argument('--feature', default='direct', type=str)
    parser.add_argument('--target_smooth_sigma', default=0.0, type=float)

    args = parser.parse_args()

    data_dir = args.dir + '/processed'
    print('Loading dataset ' + data_dir + '/data.csv')
    data_all = pandas.read_csv(data_dir + '/data.csv')
    option = TrainingDataOption(window_size=args.window, sample_step=args.step,
                                feature=args.feature, target="local_speed_gravity")
    # Create a small sample for testing
    N = data_all.shape[0]
    imu_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

    nano_to_sec = 1e09

    time_stamp = data_all['time'].values / nano_to_sec
    time_stamp -= time_stamp[0]
    time_interval = (time_stamp[1:] - time_stamp[:-1])[:, None]
    # speed_decomposed = test_decompose_speed(data_all=data_all)

    position = data_all[['pos_x', 'pos_y', 'pos_z']].values

    speed_local = compute_local_speed(time_stamp, position, orientation)

    speed_global = np.empty(speed_local.shape, dtype=float)
    for i in range(speed_local.shape[0]):
        q = quaternion.quaternion(*orientation[i])
        speed_global[i] = (q * quaternion.quaternion(1.0, *speed_local[i]) * q.conj()).vec

    print(speed_global.shape)
    position_global = np.cumsum(speed_global[:-1] * time_interval, axis=0)
    position_global += position[0]
    plt.figure('Test_local_speed')
    for i in range(3):
        plt.subplot(311 + i)
        plt.plot(time_stamp, position[:, i])
        plt.plot(time_stamp[1:], position_global[:, i])
        plt.legend(['Origin', 'Reconstructed'])
    plt.show()
