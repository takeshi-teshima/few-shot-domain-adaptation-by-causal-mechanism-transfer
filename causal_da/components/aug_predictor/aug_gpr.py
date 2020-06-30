# MLOriginalTargetDataAugGPR
"""Load different versions depending on different sklearn versions"""
from ._aug_gpr_util import MLOriginalDataAugGPR


class MLAllAugGPR:
    def __init__(self):
        raise NotImplementedError()
        pass

    def fit(self, x_train, y_train, augmenter, augment_size):
        pass

    def predict(self, x):
        return self.estimator.predict(x)


def AugGPR(HP_selection='orig_data', *args, **kwargs):
    if HP_selection == 'orig_data':
        return MLOriginalDataAugGPR(*args, **kwargs)
    elif HP_selection == 'all_data':
        return MLAllAugGPR(*args, **kwargs)
