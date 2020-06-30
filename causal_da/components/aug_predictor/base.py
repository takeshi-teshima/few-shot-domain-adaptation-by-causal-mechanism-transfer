from abc import abstractmethod


class AugPredictorBase:
    @abstractmethod
    def fit(self, x_train, y_train, augmenter, augment_size):
        """Fit the predictor (including hyper-parameter selection)."""
        raise NotImplementedError()

    def predict(self, x):
        return self.estimator.predict(x)
