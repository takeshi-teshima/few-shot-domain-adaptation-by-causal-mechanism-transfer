import numpy as np
from .metrics.emd import EMD
from .base import AugAssessmentBase


class AugEMDAssessment(AugAssessmentBase):
    def __init__(self):
        self.emd = EMD()

    def __call__(self, X_tar_tr, Y_tar_tr, X_tar_te, Y_tar_te, augmenter_output, step=None):
        X_aug_tr, Y_aug_tr, _, _ = augmenter_output
        d_truth = np.hstack((X_tar_te, Y_tar_te))
        d_aug = np.hstack((X_aug_tr, Y_aug_tr))
        metrics = {
            f'EMD between AugTarTr and TarTe': np.sum(self.emd(d_aug, d_truth)),
        }
        return metrics
