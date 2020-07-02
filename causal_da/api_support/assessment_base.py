from abc import abstractmethod


class AssessmentBase:
    """The base class to express the grouping of certain classes
    which provide the probing functionality for evaluation.
    """
    pass


class AugAssessmentBase(AssessmentBase):
    """The base class of the probing functionalities for an augmenter."""
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
    """The utility base class to express the signature of a certain type of probing functionality."""
    @abstractmethod
    def __call__(self, augmenter, x_tr, y_tr, x_te, y_te):
        pass
