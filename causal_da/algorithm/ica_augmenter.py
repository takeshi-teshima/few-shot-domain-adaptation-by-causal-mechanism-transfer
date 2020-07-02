import itertools
from math import floor
import numpy as np
import torch
import torch.nn as nn
from numpy.random import randint

# Type hinting
from typing import Union, Optional, Iterable


class ICAAugmenter(nn.Module):
    """The augmenter based on independent component analysis."""
    def __init__(self,
                 feature_extractor,
                 novelty_detector,
                 max_iter: Optional[int] = None):
        """
        Parameters:
            feature_extractor: the main feature extractor model to estimate the mixing function of ICA.
            novelty_detector: the novelty detection method that implements ``fit()``.
            max_iter: the maximum number of iterations for the ICA training.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.novelty_detector = novelty_detector
        if max_iter is None:
            self.max_iter = 100
        else:
            self.max_iter = max_iter

    def _fit_novelty_detector(self, data: np.ndarray):
        """Fit the novelty detector.

        Parameters:
            data: the data used to fit the novelty detector.
        """
        self.novelty_detector.fit(data)

    def _to_latent(self, X: np.ndarray):
        """Extract the latent values from the data.

        Parameters:
            X: input data.

        Note:
            X -> (f^{-1}) -> e
        """
        with torch.no_grad():
            e = self.feature_extractor(X)
        return e

    def _augment_latent(self, e: np.ndarray,
                        size: Optional[Union[float, int]]):
        """A utility function to generate augmented latent variables.

        Parameters:
            e: the latent representation to be augmented in the latent space.
            size: the desired size of the augmented data.

        Note:
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
                augment_size: Optional[Union[int, float]],
                with_latent: bool = False,
                include_original: bool = False,
                with_acceptance_ratio: bool = False):
        """Augment numpy arrays.

        Parameters:
            X: Data to augment
            augment_size: Size of the extension (``int``: resulting number of points. ``float``: ratio to the original data size. ``None``: full size ($n^D$ where $n$=data size, $D$=data dimension)).

        Returns:
            * xbar       (by default)
            * xbar, ebar (if with_latent == True)
        """
        e = self._to_latent(X)
        ebar = self._augment_latent(e, augment_size)
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

    def is_valid_generated_data(self, xbar) -> np.ndarray:
        """Return if data is (inside_train_data_support) and (not np.inf) and (not np.nan)

        Returns:
            an array containing the decision (``1`` if inside support. ``-1`` otherwise.)
        """
        inside = self.novelty_detector.predict(np.nan_to_num(xbar)) == 1
        not_nan = np.logical_not(np.isnan(xbar).any(axis=1))
        not_inf = np.logical_not(np.isinf(xbar).any(axis=1))
        return np.logical_and(inside, not_nan, not_inf)

    def augment_to_size(self,
                        X: np.ndarray,
                        augment_size: Optional[Union[float, int]],
                        with_latent=False,
                        include_original=False,
                        with_acceptance_ratio=False):
        """Augment data to the desired size.

        Parameters:
            X: input data of shape ``(n_data, n_dim)``
            Y: predicted data of shape ``(n_data, 1)``
            augment_size: the desired size of the augmented data.
            with_latent: whether to also return the extracted latent variables.
            include_original: whether to make sure the original data is always included in the output (in the augmented data).
            with_acceptance_ratio: whether to also return the acceptance ratio.
        """
        if augment_size is None:
            return self.augment(X,
                                augment_size,
                                with_latent=with_latent,
                                include_original=include_original,
                                with_acceptance_ratio=with_acceptance_ratio)
        else:
            return self._repeat_augment_upto_size(X, augment_size, with_latent,
                                                  include_original,
                                                  with_acceptance_ratio)

    def _repeat_augment_upto_size(self,
                                  X: np.ndarray,
                                  augment_size: Optional[Union[float, int]],
                                  with_latent=False,
                                  include_original=True,
                                  with_acceptance_ratio=False):
        """Augment numpy arrays.

        Parameters:
            X: Data to augment
            augment_size: Size of the extension. Int -> resulting number of points.
            (``float``: ratio to the original data size. ``None``: full size ($n^D$ where $n$=data size, $D$=data dimension)).

        Returns:
            * xbar       (by default)
            * xbar, ebar (if with_latent == True)
        """
        e = self._to_latent(X)
        size = get_size(X, augment_size)
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
    """Take the full product of the dimension-wise combinations of input data."""
    perms = np.array(
        list(itertools.product(range(x.shape[0]), repeat=x.shape[1])))
    return np.hstack(
        tuple(x[perms[:, d], d][:, None] for d in range(x.shape[1])))


def stochastic_combination(x: np.ndarray, size: int):
    """Take random combinations of the data values to the desired size."""
    perms = randint(x.shape[0], size=(size, x.shape[1]))
    return np.hstack(
        tuple(x[perms[:, d], d][:, None] for d in range(x.shape[1])))


def get_size(x: Iterable, size: Union[float, int]):
    """Determine the size of the augmentation."""
    if isinstance(size, int):
        n = size
    elif isinstance(size, float):
        n = floor(len(x) * size)
    else:
        n = None
    return n


class ICATransferAugmenter(ICAAugmenter):
    """A utility class to use ``ICAAugmenter`` for data with a distinction
    between the predictor variables and the predicted variables."""
    def fit(self, X_src: np.ndarray, Y_src: np.ndarray, c_src: np.ndarray,
            **kwargs):
        """Fit the model and the novelty detector.

        Parameters:
            X_src: the perdictor variables of the source domains.
            Y_src: the perdicted variables of the source domains.
            c_src: the source domain index.
        """
        data_src = np.hstack((X_src, Y_src))
        super().fit(data_src, c_src, **kwargs)
        self._fit_novelty_detector(data_src)

    def augment_to_size(self,
                        X: np.ndarray,
                        Y: np.ndarray,
                        augment_size: Optional[Union[float, int]],
                        with_latent=False,
                        include_original=True,
                        with_acceptance_ratio=False):
        """Augment data to the desired size.

        Parameters:
            X: input data of shape ``(n_data, n_dim)``
            Y: predicted data of shape ``(n_data, 1)``
            augment_size: the desired size of the augmented data.
            with_latent: whether to also return the extracted latent variables.
            include_original: whether to make sure the original data is always included in the output (in the augmented data).
            with_acceptance_ratio: whether to also return the acceptance ratio.
        """
        inputs = np.hstack((X, Y))
        result = super().augment_to_size(
            inputs,
            augment_size,
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
