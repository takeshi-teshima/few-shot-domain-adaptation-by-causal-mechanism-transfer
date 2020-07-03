from pathlib import Path
import numpy as np
import pandas as pd
from causal_da.api_support.evaluator import AugmenterEvaluatorBase
from causal_da.api_support.logging.model_logger import MLFlowMultiModelLogger

# Type hinting
from typing import Iterable, List, Dict, Optional, Union, Tuple
from causal_da.api_support.assessment_base import AugAssessmentBase, StandardAssessmentBase


class AugmentingMultiAssessmentEvaluator(AugmenterEvaluatorBase):
    """Evaluate an augmenter by running it once on a data and probing multiple assessment metrics."""
    def __init__(self,
                 x_tr: np.ndarray,
                 y_tr: np.ndarray,
                 x_te: np.ndarray,
                 y_te: np.ndarray,
                 metrics: Iterable[AugAssessmentBase],
                 augment_size: Optional[Union[int, float]],
                 namespace: str = '',
                 run_logger=None):
        """
        Parameters:
            x_tr: target domain training data predictor variables (shape ``(n_train, n_dim)``).
            y_tr: target domain training data predicted variables (shape ``(n_train, 1)``).
            x_te: target domain test data predictor variables (shape ``(n_test, n_dim)``).
            y_te: target domain test data predicted variables (shape ``(n_test, 1)``).
            metrics: the metrics to assess the augmenter.
            augment_size: the desired size of the augmentation used for the assessment.
            namespace: the namespace of the metric to be prepended.
            run_logger: the logger to save the results of the assessments.
        """
        super().__init__(namespace, run_logger)
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        self.metrics = metrics
        self.augment_size = augment_size

    def evaluate(self, augmenter, epoch: Optional[int]):
        """Perform the evaluation.

        Parameters:
            augmenter: the augmenter.
            epoch: the epoch when this evaluation is run in the training loop (``None`` if outside a training loop).
        """
        # Output of augmenter.augment_to_size() is (_X, _Y, _e, acceptance_ratio)
        augmenter_output = augmenter.augment_to_size(
            self.x_tr,
            self.y_tr,
            with_latent=True,
            include_original=False,
            augment_size=self.augment_size,
            with_acceptance_ratio=True)
        for metric in self.metrics:
            metric_out = metric(self.x_tr, self.y_tr, self.x_te, self.y_te,
                                augmenter_output, epoch)
            if metric_out is not None:
                metric_result = {
                    self.namespace + name: value
                    for name, value in metric_out.items()
                }
                self.save_results_dict(metric_result,
                                       self.get_save_results_type(epoch))


def _average_by_key(list_of_dicts: List[Dict[str, float]]):
    """From a dictionary of lists, obtain the dict whose values are the averaged values."""
    return pd.DataFrame.from_dict(list_of_dicts).mean().to_dict()


class TargetDomainsAverageEvaluator(AugmenterEvaluatorBase):
    """Evaluate the augmenter by performing the assessment on multiple domains and taking the average."""
    def __init__(self,
                 tar_tr: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 tar_te: Tuple[np.ndarray, np.ndarray, np.ndarray],
                 assessments: Iterable[StandardAssessmentBase],
                 namespace: str = '',
                 run_logger=None):
        """
        Parameters:
            tar_tr: tuple of target domain training data containing ``(x_tr, y_tr, c_tr)``.
            tar_te: tuple of target domain testing data containing ``(x_te, y_te, c_te)``.
            assessments: the assessment objects to assess the augmenter.
            metrics: the metrics to assess the augmenter.
            namespace: the namespace of the metric to be prepended.
            run_logger: the logger to save the results of the assessments.

        Note:
            ``c_tr`` and ``c_te`` are the indicators of the domain indices (this class averages the performance on multiple domains).
        """
        super().__init__(namespace, run_logger)
        self.x_tr, self.y_tr, self.c_tr = tar_tr
        self.x_te, self.y_te, self.c_te = tar_te
        self.assessments = assessments

    def evaluate(self, augmenter, epoch):
        """Perform the evaluation.

        Parameters:
            augmenter: the augmenter.
            epoch: the epoch when this evaluation is run in the training loop (``None`` if outside a training loop).
        """
        for assessment in self.assessments:
            results = []
            for c in np.unique(self.c_te):
                x_c, y_c = self.x_te[self.c_te == c], self.y_te[self.c_te == c]
                results.append(
                    assessment(augmenter, self.x_tr, self.y_tr, x_c, y_c))
            self.save_results_dict(_average_by_key(results),
                                   self.get_save_results_type(epoch))


class ModelSavingEvaluator(AugmenterEvaluatorBase):
    """The utility class to save the augmenter into a file."""
    def __init__(self, run_logger, path: str, db_key: str):
        """
        Parameters:
            run_logger: the logger to save the results of the assessments.
            path: the path where the saved model should appear.
            db_key: the database key for which the saved model path should be recorded.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model_logger = MLFlowMultiModelLogger(path, db_key, run_logger)

    def evaluate(self, augmenter, epoch):
        """Perform the evaluation.

        Parameters:
            augmenter: the augmenter.
            epoch: the epoch when this evaluation is run in the training loop (``None`` if outside a training loop).
        """
        try:
            self.model_logger.save(augmenter, epoch)
        except Exception as e:
            print(e)
