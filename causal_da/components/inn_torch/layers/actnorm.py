"""
Adopted from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
"""
import torch
import torch.nn as nn


class ActNorm(nn.Module):
    """
    Activation Normalization
    Initialize the bias and scale with a given minibatch,
    so that the output per-channel have zero mean and unit variance for that.
    After initialization, `bias` and `logs` will be trained as parameters.
    """

    def __init__(self, dim, scale=1.):
        super().__init__()
        # register mean and scale
        size = [1, dim]
        self.register_parameter("bias", nn.Parameter(torch.zeros(*size)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(*size)))
        self.dim = dim
        self.scale = float(scale)
        self.inited = False

    def initialize_parameters(self, input):
        if not self.training:
            return
        assert input.device == self.bias.device
        with torch.no_grad():
            bias = mean(input.clone(), dim=0, keepdim=True)
            vars = mean((input.clone() - bias) ** 2, dim=0, keepdim=True)
            logs = torch.log(self.scale / (torch.sqrt(vars) + 1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.inited = True

    def forward(self, input):
        if not self.inited:
            self.initialize_parameters(input)
        # center and scale
        input = (input - self.bias) * torch.exp(self.logs)
        return input

    def inv(self, input):
        # scale and center
        input = input * torch.exp(-self.logs) + self.bias
        return input


def mean(tensor, dim=None, keepdim=False):
    if dim is None:
        # mean all dim
        return torch.mean(tensor)
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdim:
            for i, d in enumerate(dim):
                tensor.squeeze_(d - i)
        return tensor
