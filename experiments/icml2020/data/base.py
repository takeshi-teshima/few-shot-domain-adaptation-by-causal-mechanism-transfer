import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .util import standardize


class DatasetBase:
    def load_data(self, path=None):
        if path is None:
            path = self.path
        _df = pd.read_csv(filepath_or_buffer=path, sep=",")
        df = _df.drop(columns=self.UNNECESSARY_KEYS)
        X = np.array(df.drop(columns=[self.Y_KEY, self.C_KEY]), dtype=np.float32)
        Y = np.array(df[self.Y_KEY], dtype=np.float32)[:, None]
        c = np.array(df[self.C_KEY])
        return X, Y, c

    def load_src_trg(self, path=None, target_c=[]):
        X, Y, c = self.load_data(path)
        X, Y = standardize(X, Y)
        src_ind, tar_ind = np.logical_not(np.isin(c, target_c)), np.isin(c, target_c)
        X_src, Y_src, c_src = X[src_ind], Y[src_ind], c[src_ind]
        _X_tar, _Y_tar, _c_tar = X[tar_ind], Y[tar_ind], c[tar_ind]
        return X_src, Y_src, c_src, _X_tar, _Y_tar, _c_tar

    def split_target_stratified(self, _X_tar, _Y_tar, _c_tar):
        if not hasattr(self, 'train_size'):
            self.train_size = len(_X_tar.shape[1] + 2)
        results = [
            train_test_split(_X_tar[_c_tar == c],
                             _Y_tar[_c_tar == c],
                             _c_tar[_c_tar == c],
                             train_size=self.train_size) for c in np.unique(_c_tar)
        ]

        X_tar_tr, X_tar_te, Y_tar_tr, Y_tar_te = (np.vstack(tuple(item[i] for item in results))
                                                  for i in range(4))
        c_tar_tr, c_tar_te = (np.hstack(tuple(item[i] for item in results)) for i in range(4, 6))

        return X_tar_tr, Y_tar_tr, X_tar_te, Y_tar_te, c_tar_tr, c_tar_te

    def load_src_stratified_trg(self, path=None, target_c=[]):
        X_src, Y_src, c_src, _X_tar, _Y_tar, _c_tar = self.load_src_trg(path, target_c)
        X_tar_tr, Y_tar_tr, X_tar_te, Y_tar_te, c_tar_tr, c_tar_te = self.split_target_stratified(
            _X_tar, _Y_tar, _c_tar)
        return X_src, Y_src, c_src, X_tar_tr, Y_tar_tr, X_tar_te, Y_tar_te, c_tar_tr, c_tar_te

    def random_target_c(self, n_target_c, path=None):
        _, _, c = self.load_data(path)
        return np.random.choice(np.unique(c), size=n_target_c, replace=False)

    def load_stratified_targets(self, n_target_c, path=None, return_target_c=True):
        target_c = self.random_target_c(n_target_c=n_target_c, path=path)
        if return_target_c:
            return self.load_src_stratified_trg(path, target_c=target_c), target_c
        else:
            return self.load_src_stratified_trg(path, target_c=target_c)
