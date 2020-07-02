import torch
import torch.nn as nn

# Type hinting
from torch import LongTensor, FloatTensor


def random_pick_wrong_target(target: LongTensor) -> LongTensor:
    """After shuffling each row, pick the one with the index just before the target.

    Parameters:
        target: the tensor (shape ``(n_sample,)``) of auxiliary variables
                used for generalized contrastive training.

    Return:
        the randomized fake auxiliary variables to be made as negative targets in generalized
        contrastive learning (shape ``(n_sample,)``).

    Note:
        Duplicate entries in ``target`` are allowed; ``torch.unique()`` is applied.
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


LOG_LOGISTIC_LOSS = nn.SoftMarginLoss()


def binary_logistic_loss(outputs: FloatTensor, positive: bool):
    """Utility function to wrap ``torch.SoftMarginLoss``."""
    if positive:
        return LOG_LOGISTIC_LOSS(outputs, torch.ones(len(outputs)))
    else:
        return LOG_LOGISTIC_LOSS(outputs, -torch.ones(len(outputs)))
