from typing import Iterable, Optional
from .api_support.evaluators import AugmenterEvaluator
from .ica_augmenter import ICATransferAugmenter


class CausalMechanismTransfer:
    """
    Implementation of Causal Mechanism Transfer.
    Takes a trainable nonlinear ICA method object and perform transfer learning.
    """
    def __init__(
            self,
            invertible_ica_model,
            predictor_model,
            aug_max_iter: Optional[int] = None,
            augmentation_size: Optional[int] = None,
    ):
        """Build CausalMechanismTransfer object.

        Parameters
        ----------
        invertible_ica_model : object
            Trainable invertible ICA model for estimating the mechanism function.
            Required to implement ``train()`` and ``inv()``.

        predictor_model : object
            Trainable predictor model to be trained on the augmented data. Needs to implement ``fit()`` and ``predict()``.

        aug_max_iter : int or None
            The maximum number of iterations for performing the augmentation.

        augmentation_size : int or None
            The size of the augmentation. Fully augmented if ``None``.

        Returns
        ----------
        None : None
        """
        self.invertible_ica_model = invertible_ica_model
        self.augmenter = ICATransferAugmenter(self.invertible_ica_model,
                                              aug_max_iter)
        self.predictor_model = predictor_model
        self.augmentation_size = augmentation_size

    def train_and_record(
            self,
            src_data,
            ica_data,
            ica_loggers,
            ica_intermediate_evaluators,
            augmenter_evaluators: Iterable[AugmenterEvaluator] = []):
        """A version of ``train()`` that also records the intermediate information.

        Parameters
        ----------
        src_data : ``numpy.ndarray``
            The source domain data to be used for fitting the novelty detector.

        ica_data : ``tuple``
            Contains the source domain data passed to the ICA method.

        train_params : ``dict``
            Keys:

            ``device`` : The device specification (e.g., ``'gpu'``) for training in PyTorch.
        """
        for evaluator in augmenter_evaluators:
            evaluator.set_augmenter(self.augmenter)
        ica_intermediate_evaluators.append(augmenter_evaluators)

        # Fit outlier detector
        self.augmenter._fit_novelty_detector(src_data)
        return self.invertible_ica_model.train_and_record(
            ica_data, ica_loggers, ica_intermediate_evaluators)

    def train(self):
        raise NotImplementedError()
