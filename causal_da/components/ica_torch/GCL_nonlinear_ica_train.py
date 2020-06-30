import numpy as np
from ignite.engine import Engine, Events
import torch
from .gcl_model import GeneralizedContrastiveICAModel
from .trainer_util import random_pick_wrong_target, _loss_1, Log1pLoss
from .logging_util import DummyRunLogger


def GCL_nonlinear_ica_train(batch_size, max_epochs,
                            gcl_ica_model: GeneralizedContrastiveICAModel,
                            data_tensor, c_src, device, optimizer,
                            epoch_callback, run_logger):

    trainerbase = GCLTrainer(gcl_ica_model,
                             optimizer,
                             contrastive_coeff=1.,
                             balance=True,
                             device=device,
                             run_logger=run_logger)

    trainer = Engine(trainerbase)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_loss(trainer):
        trainerbase.log_training_loss(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def call_epoch_callback(trainer):
        epoch_callback(trainer.state.epoch)

    dataset = torch.utils.data.TensorDataset(data_tensor,
                                             torch.LongTensor(c_src))
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True)
    trainer.run(train_loader, max_epochs=max_epochs)


class GCLTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 contrastive_coeff=1.,
                 contrastive_loss=_loss_1,
                 balance=True,
                 device='cpu',
                 run_logger=None):
        """Train a nonlinear ICA model by generalized contrastive learning.

        Parameters
        ----------
        contrastive_coeff : float (default ``1``)
            The loss can be multiplied by this coefficient to improve numerical stability.
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
        self.model.train()
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        self.optimizer.zero_grad()
        loss = self.compute_and_backward_loss(engine, data, target)
        self.optimizer.step()
        return loss

    def compute_and_backward_loss(self, trainer, data, target):
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
        print(
            f"Epoch:{trainer.state.epoch:4d}\tTrain Loss: {trainer.state.output}"
        )
        self.run_logger.log_metrics({'contrastive': trainer.state.output},
                                    step=trainer.state.epoch)
