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


class MMDH1:
    """Compute p-value for kernel two-sample test."""
    def __init__(self,
                 kernel_function='rbf',
                 iterations=10000,
                 verbose=False,
                 random_state=None,
                 **kwargs):
        self.kwargs = {
            **dict(kernel_function=kernel_function,
                   iterations=iterations,
                   verbose=verbose,
                   random_state=random_state),
            **kwargs
        }

    def __call__(self, X, Y):
        return self._kernel_two_sample_test(X, Y, **self.kwargs)[2]

    def _compute_null_distribution(self,
                                   K,
                                   m,
                                   n,
                                   iterations=10000,
                                   verbose=False,
                                   random_state=None,
                                   marker_interval=1000):
        """Compute the bootstrap null-distribution of MMD2u.
        """
        if type(random_state) == type(np.random.RandomState()):
            rng = random_state
        else:
            rng = np.random.RandomState(random_state)

        mmd2u_null = np.zeros(iterations)
        for i in range(iterations):
            if verbose and (i % marker_interval) == 0:
                print(i),
                stdout.flush()
            idx = rng.permutation(m + n)
            K_i = K[idx, idx[:, None]]
            mmd2u_null[i] = MMD2u(K_i, m, n)

        if verbose:
            print("")

        return mmd2u_null

    def _kernel_two_sample_test(self,
                                X,
                                Y,
                                kernel_function='rbf',
                                iterations=10000,
                                verbose=False,
                                random_state=None,
                                **kwargs):
        """Compute MMD^2_u, its null distribution and the p-value of the
        kernel two-sample test.

        Note that extra parameters captured by **kwargs will be passed to
        pairwise_kernels() as kernel parameters. E.g. if
        kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1),
        then this will result in getting the kernel through
        kernel_function(metric='rbf', gamma=0.1).
        """
        median = np.median(pairwise_distances(X, Y, metric='euclidean'))
        K = pairwise_kernels(np.vstack([X, Y]),
                             metric=kernel_function,
                             gamma=1.0 / (2 * median**2),
                             **kwargs)
        mmd2u = MMD2u(K, len(X), len(Y))
        if verbose:
            print("MMD^2_u = %s" % mmd2u)
            print("Computing the null distribution.")

        mmd2u_null = self._compute_null_distribution(K,
                                                     len(X),
                                                     len(Y),
                                                     iterations,
                                                     verbose=verbose,
                                                     random_state=random_state)
        p_value = max(1.0, (mmd2u_null > mmd2u).sum()) / float(iterations)
        if verbose:
            print("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0 / iterations))

        return mmd2u, mmd2u_null, p_value
