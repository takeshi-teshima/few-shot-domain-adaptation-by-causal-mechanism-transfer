import torch
import torch.nn as nn


class AffineCouplingLayer(nn.Module):
    def __init__(self, d: int, net: nn.Module):
        """
        Params:
            d: Dimension of the _split.
            net: A neural network of type (n_sample, d) -> tuple((n_sample, D - d), (n_sample, D - d))
        """
        super().__init__()
        self.d = d
        self.net = net

    def _split(self, x):
        return x[:, :self.d], x[:, self.d:]

    def _concat(self, x_fix, x_change):
        return torch.cat([x_fix, x_change], dim=1)

    def forward(self, x):
        x_fix, x_change = self._split(x)
        s, t = self.net(x_fix)
        x_change = x_change * torch.exp(s) + t
        return self._concat(x_fix, x_change)

    def inv(self, x):
        x_fix, x_change = self._split(x)
        s, t = self.net(x_fix)
        x_change = (x_change - t) * torch.exp(-s)
        return self._concat(x_fix, x_change)


class NonexponentialAffineCouplingLayer(nn.Module):
    def __init__(self, d: int, net: nn.Module):
        """
        Params:
            d: Dimension of the _split.
            net: A neural network of type (n_sample, d) -> tuple((n_sample, D - d), (n_sample, D - d))
        """
        super().__init__()
        self.d = d
        self.net = net

    def _split(self, x):
        return x[:, :self.d], x[:, self.d:]

    def _concat(self, x_fix, x_change):
        return torch.cat([x_fix, x_change], dim=1)

    def forward(self, x):
        x_fix, x_change = self._split(x)
        s, t = self.net(x_fix)
        x_change = x_change * (1 + s) + t
        return self._concat(x_fix, x_change)

    def inv(self, x):
        x_fix, x_change = self._split(x)
        s, t = self.net(x_fix)
        x_change = (x_change - t) / (1 + s)
        return self._concat(x_fix, x_change)
