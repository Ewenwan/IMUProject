import os
from os import path as osp
import math
import sys
import argparse
import json
from os import path as osp
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import quaternion
import cv2

sys.path.append(osp.join(osp.abspath(__file__), '..'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_list', type=str)
    parser.add_argument('--val_list', type=str)

