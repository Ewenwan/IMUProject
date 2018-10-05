import os
from os import path as osp
import math
import sys
from os import path as osp
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import quaternion
import torch

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pandas

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))

from algorithms import geometry
from speed_regression import training_data as td
from dl.tcn import TemporalConvNet

_feature_column = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z', 'grav_x', 'grav_y', 'grav_z',
                   'magnet_x', 'magnet_y', 'magnet_z']

# _feature_column = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z', 'grav_x', 'grav_y', 'grav_z']
# _feature_column = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z']

_nano_to_sec = 1e09

_input_channel, _output_channel = 12, 1
_layer_channels = [24, 48, 96, 192, 192, 192, 192, 96, 48, 24]

def compute_yaw_in_tango(orientations, filter_sigma=None):
    yaw = np.empty(orientations.shape[0])
    for i in range(orientations.shape[0]):
        _, _, yaw[i] = geometry.quaternion_to_euler(*orientations[i])
    return yaw


def compute_magnet_yaw_in_tango(orientations, magnet, inlier_threshold=0.1):
    north_in_tango = np.empty(magnet.shape[0])
    for i in range(magnet.shape[0]):
        q = quaternion.quaternion(*orientations[i])
        mg = (q * quaternion.quaternion(0, *magnet[i]) * q.conj()).vec[0:2]
        north_in_tango[i] = math.atan2(mg[1], mg[0])
    median_north = np.median(north_in_tango)
    diff = np.abs(north_in_tango - median_north)
    inlier = north_in_tango[diff < inlier_threshold]
    return np.mean(inlier)


def load_datalist(path):
    with open(path) as f:
        data_list = [s.strip().split(',')[0] for s in f.readlines() if s[0] != '#']

    train_features = []
    train_targets = []
    root_dir = osp.dirname(path)
    for data in data_list:
        print('Loading ', data)
        data_all = pandas.read_csv(osp.join(root_dir, data, 'processed/data.csv'))
        ts = data_all['time'].values / _nano_to_sec
        ts -= ts[0]

        positions = data_all[['pos_x', 'pos_y', 'pos_z']].values
        orientations = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
        gyro = data_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values
        gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
        magnet = data_all[['magnet_x', 'magnet_y', 'magnet_z']].values

        # gyro_aligned = geometry.align_3dvector_with_gravity(gyro, gravity)
        # linacce_aligned = geometry.align_3dvector_with_gravity(linacce, gravity)
        # feature = np.concatenate([gyro_aligned, linacce_aligned], axis=1)
        # train_features.append(gaussian_filter1d(feature, sigma=2.0, axis=0))

        # local_speed_gv = td.compute_local_speed_with_gravity(ts, positions, orientations, gravity)[:, [2]]
        # local_speed_gv = gaussian_filter1d(local_speed_gv, sigma=30.0, axis=0)
        # train_targets.append(local_speed_gv)

        yaw_in_tango = compute_yaw_in_tango(orientations)
        north_in_tango = compute_magnet_yaw_in_tango(orientations, magnet)
        gt_yaw = np.mod(yaw_in_tango - yaw_in_tango[0] + north_in_tango, 2 * math.pi)[:, np.newaxis]
        feature = gaussian_filter1d(data_all[_feature_column].values, sigma=2.0, axis=0)
        train_features.append(feature)
        train_targets.append(gt_yaw)

    return train_features, train_targets


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0.0

    def add(self, val):
        self.sum += val
        self.count += 1

    def get_average(self):
        if self == 0:
            return 0
        return self.sum / self.count


class IMUMagnetDataset(Dataset):
    def __init__(self, path, window_size, step_size):
        self.features, self.targets = load_datalist(path)
        self.sample_ids = []
        for i, feat in enumerate(self.features):
            sample_pt = np.arange(window_size / 2, feat.shape[0], step_size, dtype=np.int)[:, np.newaxis]
            if sample_pt.shape[0] == 0:
                continue
            sample_id = np.concatenate([np.ones([sample_pt.shape[0], 1], dtype=np.int) * i, sample_pt], axis=1)
            self.sample_ids.append(sample_id)
        self.sample_ids = np.concatenate(self.sample_ids, axis=0)
        self.num_feat_channel = self.features[0].shape[1]
        self.num_target_channel = self.targets[0].shape[1]
        self.window_size = window_size

    def __getitem__(self, item):
        idx, pos = self.sample_ids[item]
        if pos < self.window_size:
            feat = np.zeros([self.window_size, self.num_feat_channel], dtype=np.float32)
            feat[-pos:, :] = self.features[idx][:pos].astype(np.float32)
            label = np.zeros([self.window_size, self.num_target_channel], dtype=np.float32)
            label[-pos:, :] = self.targets[idx][:pos].astype(np.float32)
        else:
            feat = self.features[idx][pos - self.window_size:pos].astype(np.float32)
            label = self.targets[idx][pos - self.window_size:pos].astype(np.float32)
        return feat.T, label.T

    def __len__(self):
        return self.sample_ids.shape[0]


class MagnetFusionNetwork(torch.nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, layer_channels, dropout=0.2):
        super(MagnetFusionNetwork, self).__init__()
        self.tcn = TemporalConvNet(input_channel, layer_channels, kernel_size, dropout)
        self.output_layer = torch.nn.Conv1d(layer_channels[-1], output_channel, 1)
        self.output_dropout = torch.nn.Dropout(dropout)
        self.net = torch.nn.Sequential(self.tcn, self.output_layer, self.output_dropout)
        self.init_weights()

    def forward(self, x):
        return self.net(x)

    def init_weights(self):
        self.output_layer.weight.data.normal_(0, 0.001)
        self.output_layer.bias.data.normal_(0, 0.001)


def train(args):
    if args.out_dir:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))
        if osp.exists(osp.join(args.out_dir, 'logs', 'log.txt')):
            os.remove(osp.join(args.out_dir, 'logs', 'log.txt'))

    train_dataset = IMUMagnetDataset(args.train_list, args.window_size, args.step_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    val_dataset, val_loader = None, None
    if args.val_list:
        val_dataset = IMUMagnetDataset(args.val_list, args.window_size, args.step_size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Define the network
    # layer_channels = [36] * num_layers
    network = MagnetFusionNetwork(train_dataset.num_feat_channel, train_dataset.num_target_channel,
                                  args.kernel_size, _layer_channels).to(device)

    print('Network constructed')
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)

    step = 0
    best_val_loss = np.inf
    try:
        log_file = None
        if args.out_dir:
            log_file = osp.join(args.out_dir, 'logs', 'log.txt')

        for epoch in range(args.epochs):
            epoch_losses = AverageMeter()
            log_line = ''
            for batch_id, batch in enumerate(train_loader):
                feature, target = batch
                feature, target = feature.to(device), target.to(device)

                optimizer.zero_grad()
                predicted = network(feature)
                loss = criterion(predicted, target)
                loss.backward()
                optimizer.step()
                epoch_losses.add(loss.cpu().item())

                if epoch == 0 and batch_id == 0:
                    print('Initial loss %f' % (loss.cpu().item()))
                step += 1
            print('-----------------------------')
            print('Epoch %d, average loss: %f' % (epoch, epoch_losses.get_average()))
            log_line += '%d %f' % (epoch, epoch_losses.get_average())

            if val_loader:
                print('Running validation...')

                val_losses = AverageMeter()
                for val_bid, val_batch in enumerate(val_loader):
                    val_feat, val_target = val_batch
                    val_feat, val_target = val_feat.to(device), val_target.to(device)

                    optimizer.zero_grad()
                    val_out = network(val_feat)
                    val_loss = criterion(val_out, val_target)
                    val_losses.add(val_loss.cpu().item())
                log_line += ' %f\n' % val_losses.get_average()
                print('Validation loss: %f' % (val_losses.get_average()))
                if val_losses.get_average() < best_val_loss:
                    best_val_loss = val_losses.get_average()
                    if args.out_dir:
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'loss': epoch_losses.get_average(),
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        torch.save(network.state_dict(), model_path)
                        print('Model saved to ', model_path)
            else:
                log_line += '\n'
                if args.out_dir and epoch + 1 % args.save_interval == 0:
                    model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                    torch.save({'model_state_dict': network.state_dict(),
                                'epoch': epoch,
                                'loss': epoch_losses.get_average(),
                                'optimizer_state_dict': optimizer.state_dict()}, model_path)
                    print('Model saved to ', model_path)
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(log_line)
    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training completed')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)


def test(args):
    import matplotlib.pyplot as plt
    root_dir = ''
    if args.test_path:
        data_list = [args.test_path]
    elif args.test_list:
        root_dir = osp.dirname(args.test_list)
        with open(args.test_list) as f:
            data_list = [s.strip().split(',')[0] for s in f.readlines() if s[0] != '#']
    else:
        raise ValueError('Either "test_path" or "test_list" must be specified')

    if args.out_dir and not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    device = torch.device('cpu')

    model = MagnetFusionNetwork(_input_channel, _output_channel, args.kernel_size, _layer_channels)
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    model.eval().to(device)
    print('Model %s loaded to device.' % args.model_path, device)

    for data in data_list:
        print('Loading data ', data)
        imu_all = pandas.read_csv(osp.join(root_dir, data, 'processed/data.csv'))
        ts = imu_all['time'].values / _nano_to_sec
        ts -= ts[0]

        gyro = imu_all[['gyro_x', 'gyro_y', 'gyro_z']].values
        linacce = imu_all[['linacce_x', 'linacce_y', 'linacce_z']].values
        gravity = imu_all[['grav_x', 'grav_y', 'grav_z']].values
        positions = imu_all[['pos_x', 'pos_y', 'pos_x']].values
        magnets = imu_all[['magnet_x', 'magnet_y', 'magnet_z']].values
        orientations = imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

        # gyro_aligned = geometry.align_3dvector_with_gravity(gyro, gravity)
        # linacce_aligned = geometry.align_3dvector_with_gravity(linacce, gravity)
        # features = np.concatenate([gyro_aligned, linacce_aligned], axis=1)
        # features = gaussian_filter1d(features, sigma=2.0, axis=0)
        # features = features.astype(np.float32)
        # targets = td.compute_local_speed_with_gravity(ts, positions, orientations, gravity)[:, [2]]
        # targets = gaussian_filter1d(targets, sigma=30.0, axis=0)
        # targets = targets.astype(np.float32)

        features = gaussian_filter1d(imu_all[_feature_column].values, sigma=2.0, axis=0).astype(np.float32)
        yaw_in_tango = compute_yaw_in_tango(orientations)
        north_in_tango = compute_magnet_yaw_in_tango(orientations, magnets)
        targets = np.mod(yaw_in_tango - yaw_in_tango[0] + north_in_tango, 2 * math.pi)[:, np.newaxis].astype(np.float32)

        sample_points = np.arange(args.window_size / 2, features.shape[0], args.window_size, dtype=np.int)
        sample_points = np.concatenate([sample_points, [features.shape[0]]], axis=0)

        predicted = np.ones(targets.shape, dtype=np.float32)
        for pt in sample_points:
            if pt < args.window_size:
                feat = np.zeros([args.window_size, features.shape[1]], dtype=np.float32)
                feat[-pt:, :] = features[:pt, :]
            else:
                feat = features[pt - args.window_size:pt, :]
            feat = torch.Tensor(np.expand_dims(feat.T, 0), device=device)
            pred = model(feat).cpu().detach().numpy().squeeze(0).T

            if pt < args.window_size:
                predicted[:pt, :] = pred[-pt:, :]
            else:
                predicted[pt - args.window_size:pt, :] = pred

        mse = (predicted - targets) * (predicted - targets)
        mse = np.average(mse.flatten())
        print('Prediction finished. MSE: %f' % mse)
        plt.figure("Prediction for %s" % osp.split(data)[1])
        for i in range(predicted.shape[1]):
            plt.subplot(predicted.shape[1] * 100 + 11 + i)
            plt.plot(ts, targets[:, i])
            plt.plot(ts, predicted[:, i])
            plt.legend(['GT', 'Predicted'])
        plt.text(0.5, math.pi * 2, 'MSE: %f' % mse)
        plt.tight_layout()
        if args.out_dir:
            plt.savefig(osp.join(args.out_dir, 'predicted_%s.png' % osp.split(data)[1]))
            plt.close()
        else:
            plt.show()
            plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--train_list', type=str)
    parser.add_argument('--val_list', type=str)
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument('--window_size', type=int, default=1000)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise ValueError('Mode must be one of "train" or "test"')