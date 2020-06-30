import numpy as np
from sklearn.model_selection import KFold, train_test_split
from abc import abstractmethod


class SplitBase:
    @abstractmethod
    def _split(self, indices):
        raise NotImplementedError()

    def split(self, *args):
        indices = np.arange(args[0].shape[0])
        return self._split(indices)


class SingleRandomSplit(SplitBase):
    def __init__(self, train_size):
        self.train_size = train_size

    def _split(self, indices):
        ind_train, ind_test = train_test_split(indices, train_size=self.target_train_size)
        return [(ind_train, ind_test)]


class TargetSizeKFold(SplitBase):
    def __init__(self, target_train_size):
        self.target_train_size = target_train_size

    def _split(self, indices):
        folder = KFold(len(indices) // self.target_train_size)
        return [(train, test) for test, train in folder.split(indices)]
