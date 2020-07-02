import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import euclidean_distances
from .util import Timer

# Type hinting
from typing import Optional, Iterable, Union


def median_heuristic(x: np.ndarray) -> float:
    """Compute the median heuristic for those kernels
    which depend on the Euclidean distances among the input points
    (mainly for the RBF kernel = the Gaussian kernel).

    Parameters:
        x: input data.
    """
    dist = euclidean_distances(x)
    median_d = np.median(dist[np.triu_indices_from(dist, k=1)])
    gamma = 1 / (2 * median_d**2)
    return gamma


class PartialLOOCVKRR:
    """Kernel ridge regression predictor with the analytic-form leave-one-out cross-validation (LOOCV).
    The LOOCV is performed only on part of the training data
    while the training is performed on the whole generated data.
    """
    def __init__(
            self,
            alphas: Iterable[float] = [2.**i for i in range(-10, 10 + 1, 1)],
            gammas: Union[Iterable[None], Iterable[float]] = [None]):
        """
        Parameters:
            alphas: the Iterable of regularization coefficient candidates.
            gammas: the Iterable of kernel bandwidth candidates.

        Note:
            If gammas == [None], median heuristic is used.
        """
        self.alphas = alphas
        self.gammas = gammas

    def _get_ktilde(self, K: np.ndarray, alpha: float):
        """Compute $\tilde{K} := (K^\top K + \lambda I)^{-1} K^\top$.
        """
        Ktilde = np.linalg.inv(K.T @ K + alpha * np.eye(len(K))) @ K.T
        return Ktilde

    @Timer.set(lambda t: print)
    def _loocv_score(self, K: np.ndarray, alpha: float, y: np.ndarray, L: int):
        """LOOCV score computed using only the first L points as candidates for validation points.
        $K := (\phi(x_1), \ldots, \phi(x_n))^\top$
        """
        Ktilde = self._get_ktilde(K, alpha)
        H = np.hstack((np.eye(L), np.zeros(
            (L, len(y) - L)))) - K[:L, :] @ Ktilde
        score = np.mean((H @ y / H.diagonal()[:, None])**2)
        return score

    def get_heuristic_gamma(self, x: np.ndarray):
        """Apply the median heuristic used to select the kernel bandwidth.

        Parameters:
            x: the input data of shape ``(n_sample, n_dim)`` to compute the median from.
        """
        gamma = median_heuristic(x)
        return gamma

    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            aux_x: Optional[np.ndarray] = None,
            aux_y: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None):
        """Fit the predictor.

        Parameters:
            x: the predictor variables in the original data.
            y: the predicted variable in the original data.
            aux_x: the predictor variables in the augmented data.
            aux_y: the predicted variable in the augmented data.
            sample_weight: sample weight array.
        """
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

    def get_selected_params(self) -> dict:
        """Get the hyper-parameters selected by the LOOCV."""
        return {
            'gamma': self.best_gamma,
            'best_alpha': self.best_alpha,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make a prediction.

        Parameters:
            X: input data (shape ``(n_sample, n_dim)``).
        """
        _Phi = KernelRidge(kernel='rbf', gamma=self.best_gamma)._get_kernel(
            X, self.kernel_bases)
        pred = _Phi @ self.theta
        assert len(pred) == len(X)
        return pred
