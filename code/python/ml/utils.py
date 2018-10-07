import numpy as np


class AverageMeter:
    def __init__(self, ndim):
        self.count = 0
        self.sum = np.zeros(ndim)

    def add(self, val):
        self.sum += val
        self.count += 1

    def get_average(self):
        if self == 0:
            return self.sum
        return self.sum / self.count
