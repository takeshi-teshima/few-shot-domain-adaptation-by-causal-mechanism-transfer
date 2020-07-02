from abc import abstractmethod
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from causal_da.api_support.validator.base import ValidationScorerBase
from causal_da.api_support.logging.base import DummyRunLogger
from causal_da.algorithm.ica_augmenter import ICATransferAugmenter

# Type hinting
from typing import Optional, Dict


class EvaluatorBase:
    """The base class of the evaluators.

    Note:
        Usage: Override ``evaluate`` method. Do whatever evaluation you like.
    """
    def __init__(self, namespace: str = '', run_logger=None):
        """
        Parameters:
            namespace: the namespace to be prepended to the evaluation result dictionary.
            run_logger: the logger to save the results.
        """
        self.namespace = namespace
        if run_logger is None:
            run_logger = DummyRunLogger()
        self.run_logger = run_logger

    @abstractmethod
    def evaluate(self, epoch: Optional[int]):
        """Perform the evaluation.

        Parameters:
            epoch: the epoch when this evaluation is run in the training loop (``None`` if outside a training loop).
        """
        raise NotImplementedError()

    def get_save_results_type(self, epoch: Optional[int]):
        """Get the specification string of the result type.

        Parameters:
            epoch: the epoch when this evaluation is run in the training loop (``None`` if outside a training loop).

        Note:
            The method to save the results may differ between numeric values and other data types
            depending on the ``run_logger``.
        """
        return 'metric' if epoch is not None else 'tag'

    def save_results_dict(self, results: dict, logtype: str = 'metric'):
        """Save the results using the ``run_logger``.
        Parameters:
            results: the dictionary containing the results to be stored.
            logtype: the logging type of the results depending on whether this method is run inside or outside the training loop.
        """
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
        """Perform the evaluation.

        Parameters:
            epoch: the epoch when this evaluation is run in the training loop (``None`` if outside a training loop).
        """
        with torch.no_grad():
            self.evaluate(epoch)


class AugmenterEvaluatorBase(EvaluatorBase):
    """Utility base class for evaluating the augmenter.

    Note
    ----
    ``set_augmenter()`` needs to be called before evaluation.
    """
    def set_augmenter(self, augmenter: ICATransferAugmenter):
        """Set the augmenter.

        Parameters:
            augmenter: the augmenter.
        """
        self.augmenter = augmenter

    @abstractmethod
    def evaluate(self, augmenter, epoch: Optional[int]):
        """Compute the evaluation result.

        Parameters:
            augmenter: the augmenter to be evaluated.
            epoch: the epoch when this evaluation is run in the training loop (``None`` if outside a training loop).
        """
        raise NotImplementedError()

    def __call__(self, epoch):
        """Perform the evaluation.

        Parameters:
            epoch: the epoch when this evaluation is run in the training loop (``None`` if outside a training loop).
        """
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
        """
        Parameters:
            validation_scorers_dict: the dictionary containing the scorers.
            namespace: the namespace to be prepended to the result dictionary keys.
            run_logger: the logger to save the results of the assessments.
        """
        super().__init__(namespace, run_logger)
        self.validation_scorers_dict = validation_scorers_dict

    def evaluate(self, augmenter, epoch):
        """Perform the evaluation.

        Parameters:
            epoch: the epoch when this evaluation is run in the training loop (``None`` if outside a training loop).
        """
        self.save_results_dict(
            {
                name: val_scorer(augmenter)
                for name, val_scorer in self.validation_scorers_dict.items()
            }, 'metric')


class AugmenterSavingEvaluator(AugmenterEvaluatorBase):
    """The utility evaluator class to save the augmenter model."""
    def __init__(self, model_logger, namespace='', run_logger=None):
        """
        Parameters:
            model_logger: the model logger.
            namespace: the namespace to be prepended to the result dictionary keys.
            run_logger: the logger to save the results.
        """
        super().__init__(namespace, run_logger)
        self.model_logger = model_logger

    def evaluate(self, augmenter, epoch: Optional[int]):
        """
        Parameters:
            augmenter: the augmenter model to be saved.
            epoch: the epoch when this evaluation is run in the training loop (``None`` if outside a training loop).
        """
        self.model_logger.save(augmenter)
