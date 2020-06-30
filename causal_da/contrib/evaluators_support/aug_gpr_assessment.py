from sklearn.metrics import mean_squared_error
import numpy as np
from causal_da.components.aug_predictor import AugGPR
from .base import StandardAssessmentBase


class AugGPRAssessment(StandardAssessmentBase):
    def __init__(self,
                 augment_size,
                 metrics: dict = {'MSE': mean_squared_error},
                 predictor_kwargs={'alpha': 1e-10}):
        self.augment_size = augment_size
        self.metrics = metrics
        self.predictor_kwargs = predictor_kwargs

    def __call__(self, augmenter, x_tr, y_tr, x_te, y_te):
        try:
            predictor = AugGPR(**self.predictor_kwargs)
            predictor.fit(x_tr, y_tr, augmenter, self.augment_size)
            y_pred = predictor.predict(x_te)
            return {
                'AugGPR_' + name: metric(y_te, y_pred)
                for name, metric in self.metrics.items()
            }
        except Exception as e:
            print(e)
            return {
                'AugGPR_' + name: np.nan
                for name, metric in self.metrics.items()
            }
