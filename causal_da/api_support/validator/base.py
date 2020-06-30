from abc import abstractmethod
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Callable


class ValidationScorerBase:
    @abstractmethod
    def evaluate(self, model):
        raise NotImplementedError()

    def __call__(self, model):
        model.feature_extractor.eval()
        with torch.no_grad():
            val_loss = self.evaluate(model)
        model.feature_extractor.train()
        return val_loss


class LossBasedValidationScorerBase(ValidationScorerBase):
    def __init__(self, val_loss, val_data):
        self.val_loss = val_loss
        self.val_data = val_data

    def evaluate(self, model):
        return self.val_loss(model, self.val_data)


class DummyValidationScorer(ValidationScorerBase):
    def evaluate(self, model):
        return None
