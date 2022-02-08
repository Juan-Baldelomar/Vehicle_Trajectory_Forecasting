
import numpy as np


def ADE(real, pred):
    diff_sq = (real - pred)**2
    diff_sq = np.sum(diff_sq, axis=2)
    diff_sq = np.sqrt(diff_sq)
    mean_diff = np.mean(diff_sq)
    return mean_diff


real = np.array([[[1, 2], [2,3], [2,4]], [[0, 1], [0, 2], [1,3]] ])
pred = np.array([[[1, 2], [2,4], [5,8]], [[0, 2], [3, 6], [1,3]] ])


ADE(real, pred)
