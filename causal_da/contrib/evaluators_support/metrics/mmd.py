"""
Code based on https://github.com/emanuele/kernel_two_sample_test/blob/master/kernel_two_sample_test.py
"""
import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.metrics import pairwise_distances
import numpy as np
from sys import stdout
from sklearn.metrics import pairwise_kernels


def MMD2u(K, m, n):
    """The MMD^2_u unbiased statistic.
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    return 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
        1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
        2.0 / (m * n) * Kxy.sum()


class MMD:
    """Compute MMD."""
    def __init__(self, kernel_function='rbf'):
        self.kernel_function = kernel_function

    def __call__(self, X1, X2):
        """Compute MMD^2_u, its null distribution and the p-value of the
        kernel two-sample test.

        Note that extra parameters captured by **kwargs will be passed to
        pairwise_kernels() as kernel parameters. E.g. if
        kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
        then this will result in getting the kernel through
        kernel_function(metric='rbf', gamma=0.1).
        """
        median = np.median(pairwise_distances(X1, X2, metric='euclidean'))
        K = pairwise_kernels(np.vstack([X1, X2]),
                             metric=self.kernel_function,
                             gamma=1.0 / (2 * median**2))
        return MMD2u(K, len(X1), len(X2))
