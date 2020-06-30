from .base import AugAssessmentBase


class AugmenterInfoAssessment(AugAssessmentBase):
    def __call__(self, X_tar_tr, Y_tar_tr, X_tar_te, Y_tar_te, augmenter_output, step=None):
        _X, _, _, acceptance_ratio = augmenter_output
        return {'Augmented data size': len(_X), 'Acceptance ratio': acceptance_ratio}
