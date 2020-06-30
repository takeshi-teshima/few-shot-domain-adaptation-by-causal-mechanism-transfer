from abc import abstractmethod
import copy
import numpy as np
from sklearn.metrics import mean_squared_error
from causal_da.components.aug_predictor.base import AugPredictorBase


class AugScoreBase:
    @abstractmethod
    def __call__(self, augmenter, X_train, Y_train, X_test, Y_test):
        raise NotImplementedError()


class AugSklearnScore(AugScoreBase):
    def __init__(self,
                 predictor: AugPredictorBase,
                 augment_size,
                 score=mean_squared_error):
        self.predictor = predictor
        self.augment_size = augment_size
        self.score = score

    def __call__(self, augmenter, X_train, Y_train, X_test, Y_test):
        try:
            predictor = copy.deepcopy(self.predictor)
            predictor.fit(X_train, Y_train, augmenter, self.augment_size)
            return self.score(Y_test, predictor.predict(X_test))
        except Exception as e:
            print(e)
            return np.nan
