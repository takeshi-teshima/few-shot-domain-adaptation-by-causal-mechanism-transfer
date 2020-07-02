import torch
import torch.nn as nn

# Type hinting
from typing import List


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows."""
    def __init__(self, layers: List[nn.Module]):
        """
        Parameters:
            layers: a list of layers objects to stack.
        """
        super().__init__()
        self.chain = nn.ModuleList(layers)

    def forward(self, x: torch.FloatTensor):
        """Perform forward computation.

        Parameters:
            x: the input.
        """
        for i in range(len(self.chain)):
            x = self.chain[i](x)
        return x

    def inv(self, x: torch.FloatTensor):
        """Perform inverse computation.

        Parameters:
            x: the input.
        """
        inds = range(len(self.chain))
        for i in reversed(inds):
            x = self.chain[i].inv(x)
        return x
