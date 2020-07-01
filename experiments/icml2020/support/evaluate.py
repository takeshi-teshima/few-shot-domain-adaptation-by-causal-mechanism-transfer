import numpy as np
from causal_da.components.aug_predictor import AugKRR
from causal_da.api_support.evaluator import AugmenterValidationScoresEvaluator
from causal_da.api_support.evaluator_runner import EvaluatorRunner
from causal_da.contrib.evaluators import TargetDomainsAverageEvaluator, AugmentingMultiAssessmentEvaluator, ModelSavingEvaluator
from causal_da.contrib.evaluators_support import AugmenterInfoAssessment, AugKRRAssessment
from causal_da.api_support.validator.scores import AugSklearnScore
from causal_da.api_support.validator.performance import SingleTargetDomainCVPerformanceValidationScorer


def get_augmenter_evaluators(tar_tr, tar_te, augment_size, run_logger):
    """
    Example
    -------
    >>>print(True)
    False
    """
    namespace = "epoch_model"
    X_tar_tr, Y_tar_tr, c_tar_tr = tar_tr
    X_tar_te, Y_tar_te, c_tar_te = tar_te
    return [
        AugmentingMultiAssessmentEvaluator(X_tar_tr, Y_tar_tr, X_tar_te,
                                           Y_tar_te, [
                                               AugmenterInfoAssessment(),
                                           ], augment_size, namespace,
                                           run_logger),
        TargetDomainsAverageEvaluator(tar_tr, tar_te, [
            AugKRRAssessment(augment_size),
        ], namespace, run_logger),
        AugmenterValidationScoresEvaluator(
            {
                'target_cv_KRR_MSE':
                SingleTargetDomainCVPerformanceValidationScorer(
                    X_tar_tr, Y_tar_tr, AugSklearnScore(
                        AugKRR(), augment_size)),
            }, namespace, run_logger),
        ModelSavingEvaluator(run_logger, 'epoch_model', 'epoch_model')
    ]


def get_epoch_evaluator_runner(log_interval, run_logger):
    evaluators = []
    return EvaluatorRunner(evaluators, run_interval=log_interval)
