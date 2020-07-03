from typing import Iterable, Optional
from sklearn.svm import OneClassSVM
from causal_da.api_support.evaluator import AugmenterEvaluatorBase
from .ica_augmenter import ICATransferAugmenter


class CausalMechanismTransfer:
    """
    Implementation of Causal Mechanism Transfer.
    Takes a trainable nonlinear ICA method object and perform transfer learning.
    """
    def __init__(
            self,
            trainable_invertible_ica,
            predictor_model,
            novelty_detector=OneClassSVM(nu=0.1, gamma="auto"),
            aug_max_iter: Optional[int] = None,
            augmentation_size: Optional[int] = None,
    ):
        """Build CausalMechanismTransfer object.

        Parameters
        ----------
        trainable_invertible_ica : object
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
        self.trainable_invertible_ica = trainable_invertible_ica
        self.augmenter = ICATransferAugmenter(
            self.trainable_invertible_ica.get_invertible_ica_model(),
            novelty_detector=novelty_detector,
            max_iter=aug_max_iter)
        self.predictor_model = predictor_model
        self.augmentation_size = augmentation_size

    def train_and_record(
            self,
            src_data,
            ica_data,
            ica_run_logger,
            ica_intermediate_evaluators,
            ica_final_evaluators,
            augmenter_evaluators: Iterable[AugmenterEvaluatorBase] = [],
            augmenter_final_evaluators: Iterable[AugmenterEvaluatorBase] = []):
        """A version of ``train()`` that also records the intermediate information.

        Parameters
        ----------
        src_data : ``numpy.ndarray``
            The source domain data to be used for fitting the novelty detector.

        ica_data : ``tuple``
            Contains the source domain data passed to the ICA method.

        ica_run_logger

        ica_intermediate_evaluators

        ica_final_evaluators

        """
        for evaluator in augmenter_evaluators:
            evaluator.set_augmenter(self.augmenter)
        for evaluator in augmenter_final_evaluators:
            evaluator.set_augmenter(self.augmenter)
        ica_intermediate_evaluators.extend(augmenter_evaluators)
        ica_final_evaluators.extend(augmenter_final_evaluators)

        # Fit novelty detector
        self.augmenter._fit_novelty_detector(src_data)
        return self.trainable_invertible_ica.train_and_record(
            ica_data, ica_run_logger, ica_intermediate_evaluators,
            ica_final_evaluators)
