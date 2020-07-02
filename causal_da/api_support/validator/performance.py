import numpy as np
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut
from .scores import AugScoreBase
from .base import ValidationScorerBase
from .timer import Timer

# Type hinting
from typing import Union


class SingleTargetDomainCVPerformanceValidationScorer(ValidationScorerBase):
    """The scorer class to evaluate the performance by cross-validating on a single target domain."""
    def __init__(self,
                 tar_tr_X: np.ndarray,
                 tar_tr_Y: np.ndarray,
                 score: AugScoreBase,
                 cv: Union[bool, int] = True):
        """
        Parameters:
            tar_tr_X: target domain training data predictor variables.
            tar_tr_Y: target domain training data predicted variables.
            score: a scorer to measure the quality of the prediction.
            cv: the type of the cross-validation (``True``: leave-one-out cross validation. ``k: int``: ``k``-fold cross validation).
        """
        self.X, self.Y = tar_tr_X, tar_tr_Y
        self.score = score
        if cv == True:
            self.folder = LeaveOneOut()
        elif isinstance(cv, int):
            self.folder = KFold(cv)

    @Timer.set(lambda t: print(
        '[Timer] SingleTargetDomainCVPerformanceValidationScorer took:', t.time
    ))
    def evaluate(self, augmenter):
        """Perform the evaluation.

        Parameters:
            augmenter: the augmenter to be evaluated.
        """
        scores = []
        for _, (train, test) in enumerate(self.folder.split(self.X, self.Y)):
            scores.append(
                self.score(augmenter, self.X[train], self.Y[train],
                           self.X[test], self.Y[test]))
        return np.mean(scores)
