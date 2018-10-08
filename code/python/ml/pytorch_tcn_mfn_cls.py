import os
import sys
import json
from os import path as osp
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
import math

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))

from ml.tcn import TemporalConvNet
from ml.mfn_data import load_datalist
from ml.utils import AverageMeter

_feature_column = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z', 'grav_x', 'grav_y', 'grav_z',
                   'magnet_x', 'magnet_y', 'magnet_z']

_nano_to_sec = 1e09

_input_channel, _output_channel = 12, 180
# _layer_channels = [24, 48, 96, 192, 192, 192, 192, 96, 48, 24]
_layer_channels = [24, 48, 96, 96, 48]
_feature_sigma = 2.0
_target_sigma = 10.0


class IMUMagnetDataset(Dataset):
    def __init__(self, list_path, window_size, step_size):
        with open(list_path) as f:
            data_list = [s.strip().split(',')[0] for s in f.readlines() if s[0] != '#']
        root_dir = osp.dirname(list_path)
        self.angle_step = math.pi * 2 / _output_channel
        self.features, self.targets = load_datalist(root_dir, data_list, _feature_column, 'angle_cls',
                                                    feature_sigma=_feature_sigma, target_sigma=_target_sigma,
                                                    angle_step=self.angle_step)

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
            label = np.zeros([self.window_size, self.num_target_channel], dtype=np.int)
            label[-pos:, :] = self.targets[idx][:pos].astype(np.int)
        else:
            feat = self.features[idx][pos - self.window_size:pos].astype(np.float32)
            label = self.targets[idx][pos - self.window_size:pos].astype(np.int)
        return feat.T, label.T

    def __len__(self):
        return self.sample_ids.shape[0]


class IMUMagnetSeqDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path) as f:
            data_list = [s.strip().split(',')[0] for s in f.readlines() if s[0] != '#']
        root_dir = osp.dirname(list_path)
        self.angle_step = math.pi * 2 / _output_channel
        self.features, self.targets = load_datalist(root_dir, data_list, _feature_column, 'angle_cls',
                                                    feature_sigma=_feature_sigma, target_sigma=_target_sigma,
                                                    angle_step=self.angle_step)
        self.num_feature_channel = self.features[0].shape[1]
        self.num_target_channel = self.targets[0].shape[1]

    def __getitem__(self, item):
        feat = self.features[item].astype(np.float32).T
        label = self.targets[item].astype(np.int).T
        return feat, label

    def __len__(self):
        return len(self.features)


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


def write_config(args):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f)


def train(args):
    if args.out_dir:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))
        write_config(args)

    train_dataset = IMUMagnetDataset(args.train_list, args.window_size, args.step_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker)

    val_dataset, val_loader = None, None
    if args.val_list:
        val_dataset = IMUMagnetSeqDataset(args.val_list)
        val_loader = DataLoader(val_dataset)

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')

    # Define the network
    # layer_channels = [36] * num_layers
    network = MagnetFusionNetwork(train_dataset.num_feat_channel, _output_channel, args.kernel_size,
                                  _layer_channels).to(device)

    print('Network constructed')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), args.lr)

    if args.out_dir and osp.exists(osp.join(args.out_dir, 'logs', 'log.txt')):
        os.remove(osp.join(args.out_dir, 'logs', 'log.txt'))

    step = 0
    best_val_loss = np.inf
    log_file = None
    if args.out_dir:
        log_file = osp.join(args.out_dir, 'logs', 'log.txt')
    valid_seq_len = args.window_size // 2
    try:
        for epoch in range(args.epochs):
            epoch_losses = AverageMeter(1)
            log_line = ''
            for batch_id, batch in enumerate(train_loader):
                feature, target = batch
                target = torch.Tensor.squeeze(target, dim=1)
                feature, target = feature.to(device), target.to(device)
                optimizer.zero_grad()
                predicted = network(feature)
                loss = criterion(predicted[:, :, -valid_seq_len:], target[:, -valid_seq_len:])
                loss.backward()
                optimizer.step()
                epoch_losses.add(loss.cpu().item())

                if epoch == 0 and batch_id == 0:
                    print('Initial loss %f' % (loss.cpu().item()))
                step += 1
            print('-----------------------------')
            print('Epoch {}, average loss: {}'.format(epoch, epoch_losses.get_average()))
            log_line += '{} {}'.format(epoch, epoch_losses.get_average())

            if val_loader:
                val_losses = AverageMeter(1)
                for val_bid, val_batch in enumerate(val_loader):
                    val_feat, val_target = val_batch
                    val_target = torch.Tensor.squeeze(val_target, dim=1)
                    optimizer.zero_grad()
                    val_out = network(val_feat.to(device))
                    val_loss = criterion(val_out, val_target.to(device))
                    val_losses.add(val_loss.cpu().detach().numpy())
                log_line += ' {}\n'.format(val_losses.get_average())
                avg_loss = np.average(val_losses.get_average())
                print('Validation loss: {}/{}'.format(val_losses.get_average(), avg_loss))

                if avg_loss < best_val_loss:
                    best_val_loss = avg_loss
                    if args.out_dir:
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'loss': epoch_losses.get_average(),
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
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

    device = torch.device('cpu')

    model = MagnetFusionNetwork(_input_channel, _output_channel, args.kernel_size, _layer_channels)
    m = torch.load(args.model_path)
    print(m.keys())
    model.load_state_dict(m['model_state_dict'])
    model.eval().to(device)
    print('Model %s loaded to device.' % args.model_path, device)

    angle_step = math.pi * 2 / _output_channel
    features_all, targets_all = load_datalist(root_dir, data_list, _feature_column, 'angle_cls',
                                              feature_sigma=_feature_sigma, target_sigma=_target_sigma,
                                              angle_step=angle_step)
    criterion = torch.nn.CrossEntropyLoss()
    assert len(features_all) == len(targets_all)
    for data_id in range(len(features_all)):
        feat = features_all[data_id].astype(np.float32)
        target = targets_all[data_id].astype(np.int32)
        predicted_prob = model(torch.Tensor(np.expand_dims(feat.T, axis=0)))

        target_tensor = torch.Tensor(target.T).long().to(device)
        loss = criterion(predicted_prob, target_tensor).cpu().detach().item()
        print('{}, loss: {}'.format(osp.split(data_list[data_id])[1], loss))

        predicted_cls = np.argmax(predicted_prob.cpu().detach().numpy().T, axis=1)
        plt.figure("Prediction for %s" % osp.split(data_list[data_id])[1])
        for i in range(predicted_cls.shape[1]):
            plt.subplot(predicted_cls.shape[1] * 100 + 11 + i)
            plt.plot(target[:, i])
            plt.plot(predicted_cls[:, i])
            plt.legend(['GT', 'Predicted'])
            # plt.text(0.5, math.pi * 2, 'MSE: %f' % mse[i])
        plt.tight_layout()
        if args.out_dir:
            plt.savefig(osp.join(args.out_dir, 'predicted_%s.png' % osp.split(data_list[data_id])[1]))
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
    parser.add_argument('--feature_sigma', type=float, default=2.0)
    parser.add_argument('--target_sigma', type=float, default=10.0)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=1e-03)
    parser.add_argument('--window_size', type=int, default=1000)
    parser.add_argument('--step_size', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_worker', type=int, default=1)
    parser.add_argument('--kernel_size', type=int, default=3)
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
