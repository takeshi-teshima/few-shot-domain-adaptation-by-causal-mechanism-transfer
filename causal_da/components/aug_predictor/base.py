from abc import abstractmethod

# Type hinting
from typing import Optional, Union


class AugPredictorBase:
    """The base class to express the predictor that
    takes an augmenter and performs training using it."""
    @abstractmethod
    def fit(self, x_train, y_train, augmenter,
            augment_size: Optional[Union[int, float]]):
        """Fit the predictor (including hyper-parameter selection).

        Parameters:
            x_train: the training predictor variable data.
            y_train: the training predicted variable data.
            augmenter: the augmenter.
            augment_size: the desired maximum size of the augmentation.
        """
        raise NotImplementedError()

    def predict(self, x):
        """Make a prediction.

        Parameters:
            x: input data.
        """
        return self.estimator.predict(x)
