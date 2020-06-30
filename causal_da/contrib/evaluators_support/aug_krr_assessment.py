from sklearn.metrics import mean_squared_error
from causal_da.components.aug_predictor import AugKRR
import numpy as np
from .base import StandardAssessmentBase


class AugKRRAssessment(StandardAssessmentBase):
    def __init__(self,
                 augment_size,
                 metrics: dict = {'MSE': mean_squared_error},
                 predictor_kwargs={}):
        self.augment_size = augment_size
        self.metrics = metrics
        self.predictor_kwargs = predictor_kwargs

    def __call__(self, augmenter, x_tr, y_tr, x_te, y_te):
        try:
            predictor = AugKRR()
            predictor.fit(x_tr, y_tr, augmenter, self.augment_size)
            y_pred = predictor.predict(x_te)

            return {
                **{
                    'AugKRR_' + name: metric(y_te, y_pred)
                    for name, metric in self.metrics.items()
                },
                **{
                    'AugKRR_selected' + name: val
                    for name, val in predictor.get_selected_params().items()
                }
            }
        except Exception as e:
            print(e)
            return {
                'AugKRR_' + name: np.nan
                for name, metric in self.metrics.items()
            }
