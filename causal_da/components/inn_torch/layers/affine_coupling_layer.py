import torch
import torch.nn as nn

# Type hinting
from torch import FloatTensor


class AffineCouplingLayer(nn.Module):
    """The affine coupling layer (without dimension switching)."""
    def __init__(self, d: int, net: nn.Module):
        """
        Parameters:
            d : Dimension at which the input vector is split.
            net : A neural network taking a tensor of ``(n_sample, d)`` as inputs and outputs a tuple of ``numpy.ndarrays` of form: ``tuple((n_sample, D - d), (n_sample, D - d))``.
        """
        super().__init__()
        self.d = d
        self.net = net

    def _split(self, x: FloatTensor):
        """Split the input at the predesignated dimension.

        Parameters:
            x: input tensor.
        """
        return x[:, :self.d], x[:, self.d:]

    def _concat(self, x_1: FloatTensor, x_2: FloatTensor) -> FloatTensor:
        """Concatenate the split inputs.

        Parameters:
            x_1: the first part.
            x_2: the second part.
        """
        return torch.cat([x_1, x_2], dim=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """Perform forward propagation.

        Parameters:
            x: input tensor.
        """
        x_fix, x_change = self._split(x)
        s, t = self.net(x_fix)
        x_change = x_change * torch.exp(s) + t
        return self._concat(x_fix, x_change)

    def inv(self, x: FloatTensor) -> FloatTensor:
        """Perform inverse computation.

        Parameters:
            x: input tensor.
        """
        x_fix, x_change = self._split(x)
        s, t = self.net(x_fix)
        x_change = (x_change - t) * torch.exp(-s)
        return self._concat(x_fix, x_change)


class NonexponentialAffineCouplingLayer(nn.Module):
    """An affine coupling layer without the exponential activation function."""
    def __init__(self, d: int, net: nn.Module):
        """
        Parameters:
            d : Dimension at which the input vector is split.
            net : A neural network taking a tensor of ``(n_sample, d)`` as inputs and outputs a tuple of ``numpy.ndarrays` of form: ``tuple((n_sample, D - d), (n_sample, D - d))``.
        """
        super().__init__()
        self.d = d
        self.net = net

    def _split(self, x: FloatTensor):
        """Split the input at the predesignated dimension.

        Parameters:
            x: input tensor.
        """
        return x[:, :self.d], x[:, self.d:]

    def _concat(self, x_1: FloatTensor, x_2: FloatTensor) -> FloatTensor:
        """Concatenate the split inputs.

        Parameters:
            x_1: the first part.
            x_2: the second part.
        """
        return torch.cat([x_1, x_2], dim=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """Perform forward propagation.

        Parameters:
            x: input tensor.
        """
        x_fix, x_change = self._split(x)
        s, t = self.net(x_fix)
        x_change = x_change * (1 + s) + t
        return self._concat(x_fix, x_change)

    def inv(self, x: FloatTensor) -> FloatTensor:
        """Perform inverse computation.

        Parameters:
            x: input tensor.
        """
        x_fix, x_change = self._split(x)
        s, t = self.net(x_fix)
        x_change = (x_change - t) / (1 + s)
        return self._concat(x_fix, x_change)
