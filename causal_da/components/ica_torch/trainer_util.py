import torch
import torch.nn as nn


class Log1pLoss:
    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x).log1p()


def random_pick_wrong_target(target):
    """
    After shuffling each row, pick the one with the index just before the target.
    """
    # expanded = unique_labels.expand(len(target), -1)
    # masked = expanded.flatten()[.flatten()]
    unique_labels = torch.unique(target, sorted=False)[None, :]
    n_labels = unique_labels.shape[1]
    masked = torch.masked_select(unique_labels,
                                 unique_labels != target[:, None]).view(
                                     -1, n_labels - 1)
    random_indices = torch.randint(high=n_labels - 1, size=(len(target), ))
    wrong_labels = masked[torch.arange(len(masked)), random_indices][:, None]
    return wrong_labels


BCE_LOSS = torch.nn.BCEWithLogitsLoss()
LOG_LOGISTIC_LOSS = nn.SoftMarginLoss()


def _loss_1(outputs, positive: bool):
    if positive:
        return LOG_LOGISTIC_LOSS(outputs, torch.ones(len(outputs)))
    else:
        return LOG_LOGISTIC_LOSS(outputs, -torch.ones(len(outputs)))


def _loss_2(outputs, positive: bool):
    if positive:
        return BCE_LOSS(outputs, torch.ones(len(outputs)))
    else:
        return BCE_LOSS(outputs, torch.zeros(len(outputs)))
