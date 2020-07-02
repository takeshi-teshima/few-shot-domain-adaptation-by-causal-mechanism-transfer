from abc import abstractmethod
import copy
import numpy as np
from sklearn.metrics import mean_squared_error
from causal_da.components.aug_predictor.base import AugPredictorBase


class AugScoreBase:
    """The base class of the score probes to evaluate an augmenter."""
    @abstractmethod
    def __call__(self, augmenter, X_train, Y_train, X_test, Y_test):
        raise NotImplementedError()


class AugSklearnScore(AugScoreBase):
    """A score probe template to evaluate an augmenter using a predictor with a ``scikit-learn``-like interface."""
    def __init__(self,
                 predictor: AugPredictorBase,
                 augment_size,
                 score=mean_squared_error):
        """
        Parameters:
            predictor: the predictor class.
            augment_size: the desired maximum size of the augmentation.
            score: the score metric to quantify the accuracy of the predictions.
        """
        self.predictor = predictor
        self.augment_size = augment_size
        self.score = score

    def __call__(self, augmenter, X_train: np.ndarray, Y_train: np.ndarray,
                 X_test: np.ndarray, Y_test: np.ndarray):
        """Perform the evaluation.

        Parameters:
            augmenter: the augmenter.
            X_train: the training data predictor variables (shape ``(n_sample, n_dim)``).
            Y_train: the training data predicted variables (shape ``(n_sample, 1)``).
            X_test: the testing data predictor variables (shape ``(n_sample, n_dim)``).
            Y_test: the testing data predicted variables (shape ``(n_sample, 1)``).
        """
        try:
            predictor = copy.deepcopy(self.predictor)
            predictor.fit(X_train, Y_train, augmenter, self.augment_size)
            return self.score(Y_test, predictor.predict(X_test))
        except Exception as e:
            print(e)
            return np.nan
