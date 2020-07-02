import numpy as np
from sklearn.metrics import mean_squared_error
from causal_da.components.aug_predictor import AugKRR
from causal_da.api_support.assessment_base import StandardAssessmentBase

# Type hinting
from typing import Optional


class AugKRRAssessment(StandardAssessmentBase):
    """This class evaluates the augmenter by applying kernel ridge regression (KRR)
    on the augmented data and measuring its prediction error."""
    def __init__(self,
                 augment_size: Optional[int] = None,
                 metrics: dict = {'MSE': mean_squared_error}):
        """
        Parameters:
            augment_size: the size of the augmentation.
            metrics: the dict of metrics to use for evaluating the predictions.
        """
        self.augment_size = augment_size
        self.metrics = metrics

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
