import numpy as np


def index_class_label(arr):
    _, idx = np.unique(arr, return_inverse=True)
    return idx
