from abc import abstractmethod


class AssessmentBase:
    pass


class AugAssessmentBase(AssessmentBase):
    @abstractmethod
    def __call__(self,
                 X_tar_tr,
                 Y_tar_tr,
                 X_tar_te,
                 Y_tar_te,
                 augmenter_output,
                 step=None):
        pass


class StandardAssessmentBase(AssessmentBase):
    @abstractmethod
    def __call__(self, augmenter, x_tr, y_tr, x_te, y_te):
        pass


class ResultTransformer:
    def __init__(self, assessment: AssessmentBase, score_transformer):
        self.assessment = assessment
        self.score_transformer = score_transformer

    def __call__(self, *args, **kwargs):
        return self.score_transformer(self.assessment(*args, **kwargs))
