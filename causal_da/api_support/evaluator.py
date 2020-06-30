from abc import abstractmethod
from pathlib import Path
from typing import Iterable, List, Dict
import torch
import numpy as np
import pandas as pd
from causal_da.api_support.validator.base import ValidationScorerBase
from causal_da.api_support.logging.base import DummyRunLogger
from causal_da.algorithm.ica_augmenter import ICATransferAugmenter


class EvaluatorBase:
    """Usage: Override ``evaluate`` method. Do whatever evaluation you like.
    """
    def __init__(self, namespace='', run_logger=None):
        self.namespace = namespace
        if run_logger is None:
            run_logger = DummyRunLogger()
        self.run_logger = run_logger

    @abstractmethod
    def evaluate(self, epoch):
        raise NotImplementedError()

    def get_save_results_type(self, epoch):
        return 'metric' if epoch is not None else 'tag'

    def save_results_dict(self, results, logtype='metric'):
        if logtype == 'metric':
            self.run_logger.log_metrics({
                self.namespace + ' ' + name: value
                for name, value in results.items()
            })
        elif logtype == 'tag':
            self.run_logger.set_tags({
                self.namespace + ' ' + name: value
                for name, value in results.items()
            })

    def __call__(self, epoch):
        with torch.no_grad():
            self.evaluate(epoch)


class AugmenterEvaluatorBase(EvaluatorBase):
    """Utility base class for evaluating the augmenter.

    Note
    ----
    ``set_augmenter()`` needs to be called before evaluation.
    """
    def set_augmenter(self, augmenter: ICATransferAugmenter):
        self.augmenter = augmenter

    @abstractmethod
    def evaluate(self, augmenter, epoch):
        raise NotImplementedError()

    def __call__(self, epoch):
        self.augmenter.eval()
        with torch.no_grad():
            self.evaluate(self.augmenter, epoch)
        self.augmenter.train()


class AugmenterValidationScoresEvaluator(AugmenterEvaluatorBase):
    """Evaluator to record potentially relevant validation scores.

    Note
    ----
    TODO Remove this class and integrate with the others?
    """
    def __init__(self,
                 validation_scorers_dict: Dict[str, ValidationScorerBase],
                 namespace='',
                 run_logger=None):
        super().__init__(namespace, run_logger)
        self.validation_scorers_dict = validation_scorers_dict

    def evaluate(self, augmenter, epoch):
        self.save_results_dict(
            {
                name: val_scorer(augmenter)
                for name, val_scorer in self.validation_scorers_dict.items()
            }, 'metric')
