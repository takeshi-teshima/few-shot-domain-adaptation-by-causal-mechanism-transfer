import math
import torch
import torch.nn as nn

# Type hinting
from typing import Union
from torch import FloatTensor

if hasattr(torch, 'solve'):
    _pytorch_solve = torch.solve
else:
    _pytorch_solve = torch.gesv


class InvertibleLinear(nn.Module):
    """An invertible linear flow layer (i.e., the general linear group)."""
    def __init__(self,
                 n_dim: int,
                 center: Union[str, FloatTensor] = 'identity'):
        """
        Parameters:
            n_dim: the input dimension.
            center: an offset to simplify the parametrization of the linear layer.
                    * ``'identity'``: the matrix is parametrized by a difference form the identity matrix.
                    * ``torch.FloatTensor``: the matrix is parametrized by a difference from ``center``.
                    * otherwise: the matrix is parametrized by a difference from ``center``.
        """
        super().__init__()
        self.n_dim = n_dim
        self.weight = nn.Parameter(torch.FloatTensor(n_dim, n_dim))
        if center == 'identity':
            self.register_buffer('center', torch.eye(n_dim))
        elif isinstance(center, torch.FloatTensor):
            self.register_buffer('center', center)
        else:
            self.register_buffer('center', torch.zeros_like(self.weight))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """Perform forward propagation.

        Parameters:
            x: input tensor.
        """
        return torch.matmul(x, (self.center + self.weight).t())

    def inv(self, x: FloatTensor) -> FloatTensor:
        """Perform the inverse computation in a batch.

        Parameters:
            x: input tensor.
        """
        return _pytorch_solve(x[:, :, None],
                              self.center + self.weight)[0][:, :, 0]
