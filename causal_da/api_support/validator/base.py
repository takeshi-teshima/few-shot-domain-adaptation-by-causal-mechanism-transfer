from abc import abstractmethod
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

# Type hinting
from typing import Callable


class ValidationScorerBase:
    """The base class to express the specification of a validation scorer."""
    @abstractmethod
    def evaluate(self, model):
        """Define the evaluation procedure.

        Parameters:
            model: the model to be evaluated.
        """
        raise NotImplementedError()

    def __call__(self, model):
        """Perform the evaluation.

        Parameters:
            model: the model to be evaluated.
        """
        model.feature_extractor.eval()
        with torch.no_grad():
            val_loss = self.evaluate(model)
        model.feature_extractor.train()
        return val_loss


class LossBasedValidationScorerBase(ValidationScorerBase):
    """A utility class to define a scorer by a loss function."""
    def __init__(self, val_loss, val_data):
        """
        Parameters:
            val_loss: the loss function used to evaluate the quality of the predictor.
            val_data: the data used for the evaluation.
        """
        self.val_loss = val_loss
        self.val_data = val_data

    def evaluate(self, model):
        """Perform the evaluation.

        Parameters:
            model: the model to be evaluated.
        """
        return self.val_loss(model, self.val_data)
