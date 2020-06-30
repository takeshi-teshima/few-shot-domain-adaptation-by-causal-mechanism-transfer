import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import euclidean_distances
from .krr_params import DEFAULT_HYPERPARAM_CANDIDATES, DEFAULT_HYPERPARAM_DISTRIBUTIONS
from .util import Timer


def median_heuristic(x):
    dist = euclidean_distances(x)
    median_d = np.median(dist[np.triu_indices_from(dist, k=1)])
    gamma = 1 / (2 * median_d**2)
    return gamma


class PartialLOOCVKRR:
    def __init__(self,
                 alphas=DEFAULT_HYPERPARAM_CANDIDATES['alpha'],
                 gammas=DEFAULT_HYPERPARAM_CANDIDATES['gamma']):
        """Kernel ridge regression predictor with the analytic-form LOOCV.
        The LOOCV is performed only on the original training data
        while the training is performed on the whole generated data.

        Note
        ----
        If gammas == [None], median heuristic is used.
        """
        self.alphas = alphas
        self.gammas = gammas

    def _get_ktilde(self, K, alpha):
        ## Ktilde := (K' K + \lambda I)^{-1} K'
        Ktilde = np.linalg.inv(K.T @ K + alpha * np.eye(len(K))) @ K.T
        return Ktilde

    @Timer.set(lambda t: print)
    def _loocv_score(self, K, alpha, y, L):
        """LOOCV score computed using only the first L points as candidates for validation points.
            K := (\phi(x_1), \ldots, \phi(x_n))'
        """
        Ktilde = self._get_ktilde(K, alpha)
        H = np.hstack((np.eye(L), np.zeros(
            (L, len(y) - L)))) - K[:L, :] @ Ktilde
        score = np.mean((H @ y / H.diagonal()[:, None])**2)
        return score

    def get_heuristic_gamma(self, x):
        gamma = median_heuristic(x)
        return gamma

    def fit(self, x, y, aux_x=None, aux_y=None, sample_weight=None):
        try:
            self._fit(x,
                      y,
                      aux_x=aux_x,
                      aux_y=aux_y,
                      sample_weight=sample_weight)
        except Exception as e:
            print(e)

    @Timer.set(lambda t: print(f'[fit LOOCV KRR]: {t.time}'))
    def _fit(self, x, y, aux_x=None, aux_y=None, sample_weight=None):
        """

        Parameters
        ----------
            x : (n_data, dim_x)

            y : (n_data, 1)

            aux_x, aux_y : same shapes as ``x`` and ``y``, respectively.

        Note
        ----
        Only the points in `(x, y)` are used for computing the CV score.
        """

        if sample_weight is not None:
            # Normalizing the weight for numerical stability does not alter the optimization problem.
            sqrt_sw = np.sqrt(sample_weight / np.sum(sample_weight))

        L = len(x)
        if aux_x is not None:
            all_x, all_y = np.vstack((x, aux_x)), np.vstack((y, aux_y))
        else:
            all_x, all_y = x, y

        if sample_weight is not None:
            all_y *= sqrt_sw[:, None]

        self.kernel_bases = all_x

        # Get heuristic gamma
        if self.gammas[0] is None:
            self.gammas = [self.get_heuristic_gamma(all_x)]

        cv_scores = np.empty((len(self.gammas), len(self.alphas)))
        for i, gamma in enumerate(self.gammas):
            ## K can be shared among different alphas
            K = KernelRidge(kernel='rbf', gamma=gamma)._get_kernel(all_x)
            if sample_weight is not None:
                K *= np.outer(sqrt_sw, sqrt_sw)
            for j, alpha in enumerate(self.alphas):
                cv_scores[i, j] = self._loocv_score(K, alpha, all_y, L)

        ## Record the best CV score
        self.best_cv_score = np.min(cv_scores)
        ## Determine the best CV params
        i, j = np.unravel_index(np.argmin(cv_scores), cv_scores.shape)
        self.best_gamma, self.best_alpha = self.gammas[i], self.alphas[j]

        _K = KernelRidge(kernel='rbf',
                         gamma=self.best_gamma)._get_kernel(all_x)
        _Ktilde = self._get_ktilde(_K, self.best_alpha)
        self.theta = _Ktilde @ all_y

    def get_selected_params(self):
        return {
            'gamma': self.best_gamma,
            'best_alpha': self.best_alpha,
        }

    def predict(self, X):
        _Phi = KernelRidge(kernel='rbf', gamma=self.best_gamma)._get_kernel(
            X, self.kernel_bases)
        pred = _Phi @ self.theta
        assert len(pred) == len(X)
        return pred
