import torch
from torch.optim import Adam
from .gcl_model import GeneralizedContrastiveICAModel, ComponentWiseTransformWithAuxSelection
from .GCL_nonlinear_ica_train import GCL_nonlinear_ica_train


class GCLTrainableInvertibleICAModel:
    def __init__(self, inn, dim, classifier_hidden_dim, n_label,
                 classifier_n_layer):
        """Example implementation of a invertible ICA model trainable by GCL.

        Parameters
        ----------
        inn :
            An invertible neural network model implemented by PyTorch.

        dim : ``int``
            Dimensionality of the data.

        classifier_hidden_dim : ``int``

        n_label : ``int``
            The number of the labels.

        classifier_n_layer : ``int``
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
        return self.model

    def set_train_params(self, lr, weight_decay, device, batch_size,
                         max_epochs):
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = Adam(self.model.parameters(),
                              lr=lr,
                              weight_decay=weight_decay)
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def train_and_record(self, src_data, loggers, intermediate_evaluators):
        """A version of ``train()`` that also records the intermediate information.

        Parameters
        ----------
        src_data : ``tuple`` (``(numpy.ndarray, numpy.ndarray)``)
            Contains ``(data_numpy_array, labels_numpy_array)``.
            ``data_numpy_array`` is the numpy array containing the data point values (shape: ``(n_sample, dim)``).
            ``label_numpy_array`` is the numpy array containing the labels to indicate the "auxiliary information" for GCL (shape: ``(n_sample, dim)``).

        loggers : ``tuple`` (size 2)
            Contains ``(run_logger, best_score_model_logger)``
        """
        data, labels = src_data
        data_tensor = torch.from_numpy(data).float().to(self.device)
        run_logger, best_score_model_logger = loggers

        run_logger.set_tags({'trainer': 'GCLTrainer', 'objective': 'GCL'})
        run_logger.log_params({
            'batch_size': self.batch_size,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
        })
        GCL_nonlinear_ica_train(self.batch_size, self.max_epochs, self.model,
                                data_tensor, labels, self.device,
                                self.optimizer, intermediate_evaluators,
                                run_logger)
        run_logger.end_run()
