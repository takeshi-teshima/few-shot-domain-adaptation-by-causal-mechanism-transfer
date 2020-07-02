import numpy as np


def standardize(*data):
    _data = np.hstack(data)
    mean = np.mean(_data, axis=0)
    std = np.std(_data, axis=0)
    return np.hsplit((_data - mean) / std, np.cumsum(tuple(d.shape[1] for d in data))[:-1])
