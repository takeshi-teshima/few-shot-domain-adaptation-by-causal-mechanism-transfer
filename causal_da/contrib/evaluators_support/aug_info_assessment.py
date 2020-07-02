from causal_da.api_support.assessment_base import AugAssessmentBase

# Type hinting
from typing import Optional
import numpy as np


class AugmenterInfoAssessment(AugAssessmentBase):
    """This class probes some metrics of a learned augmenter."""
    def __call__(self,
                 X_tar_tr: np.ndarray,
                 Y_tar_tr: np.ndarray,
                 X_tar_te: np.ndarray,
                 Y_tar_te: np.ndarray,
                 augmenter_output: tuple,
                 epoch: Optional[int] = None):
        """Collect and return some statistics of the augmenter.

        Parameters:
            X_tar_tr: target domain training data predictor variables (shape ``(n_train, n_dim)``).
            Y_tar_tr: target domain training data predicted variables (shape ``(n_train, 1)``).
            X_tar_te: target domain test data predictor variables (shape ``(n_test, n_dim)``).
            Y_tar_te: target domain test data predicted variables (shape ``(n_test, 1)``).
            augmenter_output: the tuple of ``(_X, _Y, _e, acceptance_ratio)``.
            epoch: the epoch at which this evaluation is run.
        """
        _X, _, _, acceptance_ratio = augmenter_output
        return {
            'Augmented data size': len(_X),
            'Acceptance ratio': acceptance_ratio
        }
