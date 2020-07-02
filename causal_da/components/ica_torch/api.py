import torch
from torch.optim import Adam
from .gcl_model import GeneralizedContrastiveICAModel, ComponentWiseTransformWithAuxSelection
from .GCL_nonlinear_ica_train import GCL_nonlinear_ica_train

# Type hinting
from typing import Tuple
import numpy as np
import torch.nn as nn
from causal_da.api_support.evaluator_runner import EvaluatorRunner


class GCLTrainableInvertibleICAModel:
    """Example implementation of a invertible ICA model trainable by GCL."""
    def __init__(self, inn: nn.Module, dim: int, classifier_hidden_dim: int,
                 n_label: int, classifier_n_layer: int):
        """
        Parameters:
            inn : invertible neural network model implemented by PyTorch.
            dim : dimensionality of the data.
            classifier_hidden_dim : the number of hidden units used to construct the classifier model in the wrapper class for training via GCL.
            n_label : the cardinality of the label candidates.
            classifier_n_layer : the depth of the classifier wrapper class for GCL training.
        """
        self.inn = inn
        self.model = GeneralizedContrastiveICAModel(
            self.inn,
            dim,
            n_label,
            componentwise_transform=ComponentWiseTransformWithAuxSelection(
                dim,
                n_label,
                hidden_dim=classifier_hidden_dim,
                n_layer=classifier_n_layer))

    def get_invertible_ica_model(self):
        """Utility method to obtain the classifier (wrapper) model."""
        return self.model

    def set_train_params(self, lr: float, weight_decay: float, device: str,
                         batch_size: int, max_epochs: int):
        """
        Parameters:
            lr: the learning rate.
            weight_decay: the weight decay.
            device: the processing unit (GPU or CPU) identifier to run the training on (``'cpu'``: run on CPU).
            batch_size: the batch size.
            max_epochs: the maximum number of epochs to train.
        """
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = Adam(self.model.parameters(),
                              lr=lr,
                              weight_decay=weight_decay)
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def train_and_record(self, src_data: Tuple[np.ndarray, np.ndarray],
                         run_logger, intermediate_evaluators: EvaluatorRunner,
                         final_evaluators: EvaluatorRunner):
        """A version of ``train()`` that also records the intermediate information.

        Parameters:
            src_data : Contains ``(data_numpy_array, labels_numpy_array)``.

                       * ``data_numpy_array`` is the numpy array containing the data point values (shape: ``(n_sample, dim)``).
                       * ``label_numpy_array`` is the numpy array containing the labels to indicate the "auxiliary information" for GCL (shape: ``(n_sample, dim)``).

            run_logger: the logger to save the results.

            intermediate_evaluators: the evaluators to be called at the end of every epoch.

            final_evaluators: the evaluators to be called at the end of the training.
        """
        data, labels = src_data
        data_tensor = torch.from_numpy(data).float().to(self.device)

        run_logger.set_tags({'trainer': 'GCLTrainer', 'objective': 'GCL'})
        run_logger.log_params({
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'device': self.device,
        })
        GCL_nonlinear_ica_train(data_tensor, labels, self.batch_size,
                                self.max_epochs, self.model, self.device,
                                self.optimizer, intermediate_evaluators,
                                final_evaluators, run_logger)
        run_logger.end_run()
