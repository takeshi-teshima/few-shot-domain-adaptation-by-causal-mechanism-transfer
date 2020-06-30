import numpy as np
from sklearn.model_selection import KFold, train_test_split, LeaveOneOut
from .scores import AugScoreBase
from .base import ValidationScorerBase
from .util import SingleRandomSplit, TargetSizeKFold
from .timer import Timer


class SingleTargetDomainCVPerformanceValidationScorer(ValidationScorerBase):
    def __init__(self, tar_tr_X, tar_tr_Y, score: AugScoreBase, cv=True):
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
        scores = []
        for _, (train, test) in enumerate(self.folder.split(self.X, self.Y)):
            scores.append(
                self.score(augmenter, self.X[train], self.Y[train],
                           self.X[test], self.Y[test]))
        return np.mean(scores)
