import numpy as np


def index_class_label(arr: np.ndarray):
    """Turn the class label identifiers into numeric indices."""
    _, idx = np.unique(arr, return_inverse=True)
    return idx
