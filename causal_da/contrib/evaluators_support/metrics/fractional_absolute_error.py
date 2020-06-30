import numpy as np


def fractional_absolute_error(y, y_pred):
    """https://pytorch.org/ignite/contrib/metrics.html#ignite.contrib.metrics.regression.FractionalAbsoluteError"""
    return (2 * np.abs(y - y_pred) / (np.abs(y_pred) + np.abs(y))).sum() / y.shape[0]
                                                                                                                   