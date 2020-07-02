import numpy as np
from ignite.engine import Engine, Events
import torch
from .gcl_model import GeneralizedContrastiveICAModel
from .trainer_util import random_pick_wrong_target, binary_logistic_loss
from .logging_util import DummyRunLogger

# Type hinting
from typing import Callable
from torch import FloatTensor, LongTensor
BinaryCallableLoss = Callable[[FloatTensor, int], FloatTensor]


def GCL_nonlinear_ica_train(data_tensor: FloatTensor, c_src: LongTensor,
                            batch_size: int, max_epochs: int,
                            gcl_ica_model: GeneralizedContrastiveICAModel,
                            device: str, optimizer, epoch_callback,
                            final_callback, run_logger):
    """Perform generalized contrastive learning (GCL) for nonlinear independent component analysis (nonlinear ICA).

    Parameters:
        data_tensor: the training data input variables (shape ``(n_sample,)``).
        c_src: the auxiliary variable used as labels in the contrastive learning (shape ``(n_sample,)``).
        batch_size: the batch size for training.
        max_epochs: the maximum number of epochs to run the training.
        gcl_ica_model: the ICA model that can be trained via GCL.
        device: the device identifier (``'cpu'``: use CPU).
        optimizer: the ``pytorch`` optimizer.
        epoch_callback: The callback to be called after every epoch the training loop.
        final_callback: The callback to be called at the end of the training loop.
                        To be called with the single argument ``None``.
        run_logger: the logger to save the results.
    """

    trainerbase = GCLTrainer(gcl_ica_model,
                             optimizer,
                             contrastive_coeff=1.,
                             balance=True,
                             device=device,
                             run_logger=run_logger)

    trainer = Engine(trainerbase)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        """Callback at the end of each epoch to record the training loss."""
        trainerbase.log_training_loss(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def call_epoch_callback(trainer):
        """User-defined callback at the end of each epoch."""
        epoch_callback(trainer.state.epoch)

    @trainer.on(Events.COMPLETED)
    def call_final_callback(_):
        """User-defined callback at the end of the training process."""
        final_callback(None)

    dataset = torch.utils.data.TensorDataset(data_tensor,
                                             torch.LongTensor(c_src))
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)
    trainer.run(train_loader, max_epochs=max_epochs)


class GCLTrainer:
    """Trainer class conforming to the interface of the ``torch-ignite`` package."""
    def __init__(self,
                 model,
                 optimizer,
                 contrastive_coeff: float = 1.,
                 contrastive_loss: BinaryCallableLoss = binary_logistic_loss,
                 balance: bool = True,
                 device: str = 'cpu',
                 run_logger=None):
        """Train a nonlinear ICA model by generalized contrastive learning.

        Parameters:
            model: the model to be trained.
            optimizer: the optimizer.
            contrastive_coeff: The loss can be multiplied by this coefficient to improve numerical stability.
            contrastive_loss: the loss function used for contrastive training.
            balance: whether to use a coefficient to stabilize the learning (the coefficient is initialized to normalize the loss to 1 in the first iteration).
            device: whether to use GPU for training.
                    * ``gpu_identifier``: use GPU with the identifier if available.
                    * ``'cpu'``: use CPU.
            run_logger: the logger to save the results.
        """
        self.contrastive_coeff = contrastive_coeff
        self.contrastive_loss = contrastive_loss
        self.balance = balance
        if run_logger is None:
            self.run_logger = DummyRunLogger()
        else:
            self.run_logger = run_logger
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def __call__(self, engine, batch):
        """Perform one training iteration by back-propagation using the optimizer."""
        self.model.train()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        self.optimizer.zero_grad()
        loss = self.compute_and_backward_loss(engine, data, target)
        self.optimizer.step()
        return loss

    def compute_and_backward_loss(self, trainer, data: FloatTensor,
                                  target: LongTensor):
        """Compute loss and prepare the back-propagation.

        Parameters:
            trainer: the trainer.
            data: the input data (shape ``(n_sample, n_dim)``).
            target: the auxiliary variables for GCL (shape ``(n_sample,)``).
        """
        pos_output = self.model.classify((data, target[:, None]),
                                         return_hidden=False)
        negative_targets = random_pick_wrong_target(target)
        neg_output = self.model.classify((data, negative_targets),
                                         return_hidden=False)
        contrastive_term = self.contrastive_loss(
            pos_output, True) + self.contrastive_loss(neg_output, False)

        # For numerical stability.
        if (trainer.state.epoch == 1) and (trainer.state.iteration == 1):
            if self.balance:
                self.scale_contrastive_term = contrastive_term.item()
            else:
                self.scale_contrastive_term = 1

        loss = self.contrastive_coeff * contrastive_term / self.scale_contrastive_term
        loss.backward()
        return tuple(l.item() for l in (contrastive_term, ))

    def log_training_loss(self, trainer):
        """Record the training loss metrics.

        Parameters:
            trainer: the trainer object.
        """
        print(
            f"Epoch:{trainer.state.epoch:4d}\tTrain Loss: {trainer.state.output}"
        )
        self.run_logger.log_metrics({'contrastive': trainer.state.output},
                                    step=trainer.state.epoch)
