import numpy as np
import pandas
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import argparse
from os import path as osp
import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
import cv2

from ml.mfn_data import compute_north_in_tango


def map_overlay(img, positions, offset, scale, yaw, ref_dir=None):
    img2 = np.copy(img)
    yaw -= math.pi / 2
    rotor = np.array([[math.cos(-yaw), -math.sin(-yaw)], [math.sin(-yaw), math.cos(-yaw)]])
    positions_vis = np.matmul(rotor, positions.T * scale).T
    positions_vis[:, 1] *= -1
    positions_vis += offset
    for i in range(positions_vis.shape[0]):
        cv2.circle(img2, (int(positions_vis[i][0]), int(positions_vis[i][1])), 1, (255, 0, 0))
    if ref_dir is not None:
        ref_dir = np.matmul(rotor, ref_dir.T).T
        ref_dir[1] *= -1
        north_len = 40.0
        vtx1 = np.array([img2.shape[1] / 2, img2.shape[0] / 2]).astype(np.int)
        vtx2 = (vtx1 + ref_dir * north_len).astype(np.int)
        cv2.line(img2, (vtx1[0], vtx1[1]), (vtx2[0], vtx2[1]), (0, 0, 255), 3)
        cv2.circle(img2, (vtx1[0], vtx1[1]), 5, (255, 0, 0), 5)
    return img2


class AlignmentVisualizer(object):
    def __init__(self, path, ref_img, window_name='Alignment', fig_size=(10, 10)):
        imu_all = pandas.read_csv(osp.join(path, 'processed/data.csv'))
        self.path = path
        self.positions = imu_all[['pos_x', 'pos_y', 'pos_z']].values
        self.orientations = imu_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
        self.magnets = imu_all[['magnet_x', 'magnet_y', 'magnet_z']].values
        self.ref_img = ref_img
        self.img_h, self.img_w, _ = ref_img.shape

        # Properly rescale the trjectory such that it fits the image.
        self.positions -= np.average(self.positions, axis=0)
        self.positions = self.positions[::5, :2]
        max_radius = np.amax(np.linalg.norm(self.positions, axis=1), axis=0)
        self.positions = self.positions * 0.8 / max_radius * min(self.img_w / 2, self.img_h / 2)
        self.offset = np.array([self.img_w / 2, self.img_h / 2])
        self.scale = 1.0
        self.yaw = compute_north_in_tango(self.orientations, self.magnets)
        rotor = np.array([[math.cos(self.yaw), -math.sin(self.yaw)], [math.sin(self.yaw), math.cos(self.yaw)]])
        self.ref_dir = np.matmul(rotor, np.array([1.0, 0.0])).T

        plt.figure(window_name, fig_size)
        self.img_axes = plt.axes([0, 0.2, 1., 0.75])
        self.update()

        self.widgets = {}
        self._setup_widgets()

        plt.show()

    def __del__(self):
        plt.close()

    def _slider_callback_pos_x(self, val):
        self.offset[0] = val
        self.update()

    def _slider_callback_pos_y(self, val):
        self.offset[1] = val
        self.update()

    def _slider_callback_scale(self, val):
        self.scale = val
        self.update()

    def _slider_callback_yaw(self, val):
        self.yaw = val
        self.update()

    def _btn_callback_save(self, event):
        cv2.imwrite(osp.join(self.path, 'processed/map.png'), self.ref_img)
        out_path = osp.join(self.path, 'processed/map_alignment.txt')
        with open(out_path, 'w') as f:
            f.write('{} {} {} {}\n'.format(self.offset[0], self.offset[1], self.scale, self.yaw))
        print('Alignment saved to {}'.format(out_path))

    def _setup_widgets(self):
        self.widgets['sld_pos_x'] = Slider(plt.axes([0.1, 0.125, 0.8, 0.025]), 'Pos_x', 0, self.img_w, self.img_w / 2)
        self.widgets['sld_pos_x'].on_changed(self._slider_callback_pos_x)

        self.widgets['sld_pos_y'] = Slider(plt.axes([0.1, 0.095, 0.8, 0.025]), 'Pos_y', 0, self.img_h, self.img_h / 2)
        self.widgets['sld_pos_y'].on_changed(self._slider_callback_pos_y)

        self.widgets['sld_scale'] = Slider(plt.axes([0.1, 0.065, 0.8, 0.025]), 'Scale', 0.1, 10.0, 1.0)
        self.widgets['sld_scale'].on_changed(self._slider_callback_scale)

        self.widgets['sld_yaw'] = Slider(plt.axes([0.1, 0.035, 0.8, 0.025]), "Yaw", -math.pi, math.pi, self.yaw)
        self.widgets['sld_yaw'].on_changed(self._slider_callback_yaw)

        self.widgets['btn_save'] = Button(plt.axes([0.65, 0.005, 0.1, 0.02]), "Save")
        self.widgets['btn_save'].on_clicked(self._btn_callback_save)

        self.widgets['btn_exit'] = Button(plt.axes([0.8, 0.005, 0.1, 0.02]), "Exit")
        self.widgets['btn_exit'].on_clicked(lambda _: plt.close())

    def update(self):
        vis_img = map_overlay(self.ref_img, self.positions, self.offset, self.scale, self.yaw, self.ref_dir)
        self.img_axes.imshow(vis_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--ref_path', type=str)
    parser.add_argument('--downsample', type=float, default=0.25)

    args = parser.parse_args()
    ref_img = cv2.imread(args.ref_path)

    visualier = AlignmentVisualizer(args.path, ref_img)
