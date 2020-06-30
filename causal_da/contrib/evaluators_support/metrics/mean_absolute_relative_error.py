import numpy as np


def mean_absolute_relative_error(y, y_pred):
    """https://pytorch.org/ignite/contrib/metrics.html#ignite.contrib.metrics.regression.FractionalAbsoluteError"""
    return (np.abs(y - y_pred) / np.abs(y)).sum() / y.shape[0]
                                                                                                                   