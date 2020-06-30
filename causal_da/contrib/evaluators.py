from pathlib import Path
from typing import Iterable, List, Dict
import numpy as np
from causal_da.api_support.evaluator import AugmenterEvaluatorBase
from causal_da.api_support.logging.model_logger import MLFlowMultiModelLogger


class AugmentingMultiAssessmentEvaluator(AugmenterEvaluatorBase):
    def __init__(self,
                 x_tr,
                 y_tr,
                 x_te,
                 y_te,
                 metrics,
                 augment_size,
                 namespace='',
                 run_logger=None):
        super().__init__(namespace, run_logger)
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_te = x_te
        self.y_te = y_te
        self.metrics = metrics
        self.augment_size = augment_size

    def evaluate(self, epoch):
        # _X, _Y, _e, acceptance_ratio
        augmenter_output = self.augmenter.augment_to_size(
            self.x_tr,
            self.y_tr,
            with_latent=True,
            include_original=False,
            size=self.augment_size,
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
    return pd.DataFrame.from_dict(list_of_dicts).mean().to_dict()


class TargetDomainsAverageEvaluator(AugmenterEvaluatorBase):
    def __init__(self,
                 tar_tr,
                 tar_te,
                 assessments,
                 namespace='',
                 run_logger=None):
        super().__init__(namespace, run_logger)
        self.x_tr, self.y_tr, self.c_tr = tar_tr
        self.x_te, self.y_te, self.c_te = tar_te
        self.assessments = assessments

    def evaluate(self, augmenter, epoch):
        for assessment in self.assessments:
            results = []
            for c in np.unique(self.c_te):
                x_c, y_c = self.x_te[self.c_te == c], self.y_te[self.c_te == c]
                results.append(
                    assessment(augmenter, self.x_tr, self.y_tr, x_c, y_c))
            self.save_results_dict(_average_by_key(results),
                                   self.get_save_results_type(epoch))


class ModelSavingEvaluator(AugmenterEvaluatorBase):
    def __init__(self, run_logger, path: str, db_key: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model_logger = MLFlowMultiModelLogger(path, db_key, run_logger)

    def __call__(self, augmenter, epoch):
        try:
            self.model_logger.save(augmenter, epoch)
        except Exception as e:
            print(e)
