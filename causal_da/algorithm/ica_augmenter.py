import itertools
from itertools import permutations
from math import floor
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.random import randint
from sklearn.svm import OneClassSVM


class ICAAugmenter(nn.Module):
    def __init__(self,
                 feature_extractor,
                 novelty_detector=OneClassSVM(nu=0.1, gamma="auto"),
                 max_iter=100):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.novelty_detector = novelty_detector
        self.max_iter = max_iter

    def _fit_novelty_detector(self, data):
        self.novelty_detector.fit(data)

    def _to_latent(self, X: np.ndarray):
        """X -> (f^{-1}) -> e"""
        with torch.no_grad():
            e = self.feature_extractor(X)
        return e

    def _augment_latent(self, e: np.ndarray, size):
        """A utility function to generate augmented latent variables.
        e -> (independentify) -> ebar
        """
        if size is None:
            ebar = full_combination(e)
        else:
            size = get_size(e, size)
            ebar = stochastic_combination(e, size)
        return ebar

    def augment(self,
                X: np.ndarray,
                size,
                with_latent=False,
                include_original=False,
                with_acceptance_ratio=False):
        """Augment numpy arrays.

        Parameters
        ----------
        X:
            Data to augment

        size : (int, float, or None)
            Size of the extension.

            ``int`` -> resulting number of points.

            ``float`` -> ratio to the original data size.

            ``None`` -> Full size ($n^D$ where $n$=data size, $D$=data dimension).

        Returns
        ----------
            xbar       (by default)

            xbar, ebar (if with_latent == True)
        """
        e = self._to_latent(X)
        ebar = self._augment_latent(e, size)
        with torch.no_grad():
            xbar = self.feature_extractor.inv(ebar)  # ebar -> (f) -> xbar
        inside_train_data_support = self.is_valid_generated_data(xbar)
        xbar, ebar = xbar[inside_train_data_support], ebar[
            inside_train_data_support]
        if include_original:
            xbar, ebar = np.vstack((xbar, X)), np.vstack((ebar, e))
        _res = [xbar]
        if with_latent:
            _res.append(ebar)
        if with_acceptance_ratio:
            _res.append(np.mean(inside_train_data_support))
        return tuple(_res)

    def is_valid_generated_data(self, xbar):
        """Return if data is (inside_train_data_support) and (not np.inf) and (not np.nan)

        Returns
        -------
            ``np.ndarray``: ``1`` if inside support, ``-1`` otherwise
        """
        inside = self.novelty_detector.predict(np.nan_to_num(xbar)) == 1
        not_nan = np.logical_not(np.isnan(xbar).any(axis=1))
        not_inf = np.logical_not(np.isinf(xbar).any(axis=1))
        return np.logical_and(inside, not_nan, not_inf)

    def augment_to_size(self,
                        X: np.ndarray,
                        size: Optional[Union[float, int]],
                        with_latent=False,
                        include_original=False,
                        with_acceptance_ratio=False):
        if size is None:
            return self.augment(X,
                                size,
                                with_latent=with_latent,
                                include_original=include_original,
                                with_acceptance_ratio=with_acceptance_ratio)
        else:
            return self._repeat_augment_upto_size(X, size, with_latent,
                                                  include_original,
                                                  with_acceptance_ratio)

    def _repeat_augment_upto_size(self,
                                  X: np.ndarray,
                                  size: Union[float, int],
                                  with_latent=False,
                                  include_original=True,
                                  with_acceptance_ratio=False):
        """Augment numpy arrays.

        Params:
            X: Data to augment

            size (int, float, or None): Size of the extension. Int -> resulting number of points. Float -> ratio to the original data size. None -> Full size ($n^D$ where $n$=data size, $D$=data dimension).

        Return:
            xbar       (by default)

            xbar, ebar (if with_latent == True)
        """
        e = self._to_latent(X)
        size = get_size(X, size)
        _acceptance_ratios = []
        if include_original:
            ret, ret_e = X, e
        else:
            ret, ret_e = np.empty((0, X.shape[1])), np.empty((0, e.shape[1]))
        for _ in range(self.max_iter):
            if len(ret) > size:
                break
            _ebar = self._augment_latent(e, size)
            with torch.no_grad():
                _xbar = self.feature_extractor.inv(
                    _ebar)  # ebar -> (f) -> xbar
            accepted = self.is_valid_generated_data(_xbar)
            _acceptance_ratios.append(accepted.sum() / len(_ebar))
            xbar, ebar = _xbar[accepted], _ebar[accepted]
            ret, ret_e = np.vstack((ret, xbar)), np.vstack((ret_e, ebar))
            assert not np.logical_or(
                np.isinf(xbar).any(axis=1),
                np.isnan(xbar).any(axis=1)).any()
        _res = [ret[:size]]
        if with_latent:
            _res.append(ret_e[:size])
        if with_acceptance_ratio:
            _res.append(np.mean(_acceptance_ratios))
        return tuple(_res)


def full_combination(x: np.ndarray):
    perms = np.array(
        list(itertools.product(range(x.shape[0]), repeat=x.shape[1])))
    return np.hstack(
        tuple(x[perms[:, d], d][:, None] for d in range(x.shape[1])))


def stochastic_combination(x: np.ndarray, size: int):
    perms = randint(x.shape[0], size=(size, x.shape[1]))
    return np.hstack(
        tuple(x[perms[:, d], d][:, None] for d in range(x.shape[1])))


def get_size(x, size):
    if isinstance(size, int):
        n = size
    elif isinstance(size, float):
        n = floor(len(x) * size)
    else:
        n = None
    return n


class ICATransferAugmenter(ICAAugmenter):
    def fit(self, X_src, Y_src, c_src, **kwargs):
        data_src = np.hstack((X_src, Y_src))
        super().fit(data_src, c_src, **kwargs)
        self._fit_novelty_detector(data_src)

    def augment_to_size(self,
                        X: np.ndarray,
                        Y: np.ndarray,
                        size: Union[float, int],
                        with_latent=False,
                        include_original=True,
                        with_acceptance_ratio=False):
        inputs = np.hstack((X, Y))
        result = super().augment_to_size(
            inputs,
            size,
            with_latent=with_latent,
            include_original=include_original,
            with_acceptance_ratio=with_acceptance_ratio)
        if with_latent and with_acceptance_ratio:
            xbar, ebar, acceptance_ratio = result
            return xbar[:, :-1], xbar[:, -1][:, None], ebar, acceptance_ratio
        elif with_latent:
            xbar, ebar = result
            return xbar[:, :-1], xbar[:, -1][:, None], ebar
        else:
            xbar, = result
            return xbar[:, :-1], xbar[:, -1][:, None]
