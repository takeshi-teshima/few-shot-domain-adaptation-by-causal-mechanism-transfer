import torch
import torch.nn as nn


class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super().__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x):
        for i in range(len(self.chain)):
            x = self.chain[i](x)
        return x

    def inv(self, x):
        inds = range(len(self.chain))
        for i in reversed(inds):
            x = self.chain[i].inv(x)
        return x
