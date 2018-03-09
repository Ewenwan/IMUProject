import matplotlib.pyplot as plt
import os
import math
import numpy as np
import quaternion
import pandas
import cv2
from scipy.ndimage.filters import maximum_filter1d
from scipy.signal import lfilter, kaiserord, firwin
from scipy import interpolate

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


