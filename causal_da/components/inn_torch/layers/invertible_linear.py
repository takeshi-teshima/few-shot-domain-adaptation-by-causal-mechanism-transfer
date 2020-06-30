import math
import torch
import torch.nn as nn

if hasattr(torch, 'solve'):
    pytorch_solve = torch.solve
else:
    pytorch_solve = torch.gesv


class InvertibleLinear(nn.Module):
    def __init__(self, n_dim, center='identity'):
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

    def forward(self, x):
        return torch.matmul(x, (self.center + self.weight).t())

    def inv(self, x):
        """Batch solve inverse.
        """
        return pytorch_solve(x[:, :, None], self.center + self.weight)[0][:, :, 0]
