import torch
import torch.nn as nn
import torch.nn.functional as F
import inn_torch.layers as layers
from inn_torch.wrappers import SequentialFlow

# Type hinting
from torch import FloatTensor


class GlowNonExponential(nn.Module):
    """A Glow model based on the exponential-activation-free affine coupling layers."""
    def __init__(self, depth: int, dim: int, n_hidden: int = 20):
        """
        Parameters:
            depth: The number of the (invertible linear, affine coupling) flow layer pairs to stack.
            dim: The dimension of the input data.
            n_hidden: The number of the hidden units of the parametrization of $s$ and $t$ in each affine coupling layer (the number of layers is fixed at one-hidden-layer).

        Note:
            Since we employ invertible linear layers, we do not require
            dimension-swap layers in between the affine coupling layers.
        """
        super().__init__()
        chain = []
        D, d = dim, dim // 2
        chain.append(layers.ActNorm(dim))
        for _ in range(depth):
            chain.append(layers.InvertibleLinear(dim))
            chain.append(
                layers.NonexponentialAffineCouplingLayer(
                    d, _NN(d, n_hidden, D - d)))
        self.net = SequentialFlow(chain)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """Perform forward propagation.

        Parameters:
            x: input tensor.
        """
        return self.net(x)

    def inv(self, x: FloatTensor) -> FloatTensor:
        """Perform the inverse computation in a batch.

        Parameters:
            x: input tensor.
        """
        return self.net.inv(x)

    def randomize_weights(self):
        """Perform random initialization of the trainable parameters."""
        for net in self.net.chain:
            if isinstance(net, layers.AffineCouplingLayer):
                net.net.randomize_weights()
        return self


class _NN(nn.Module):
    """A utility neural network model (one-hidden-layer network) for affine coupling layers."""
    def __init__(self,
                 n_input: int,
                 n_hidden: int,
                 n_out: int,
                 scale: bool = False):
        """
        Parameters:
            n_input: the input dimension.
            n_hidden: the number of the hidden units.
            n_out: the output dimension.
            scale: whether to train an extra coefficient on the output of $s$.
        """
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc_s = nn.Linear(n_hidden, n_out)
        self.fc_t = nn.Linear(n_hidden, n_out)

        nn.init.constant_(self.fc1.weight, 0.0)
        nn.init.constant_(self.fc_s.weight, 0.0)
        nn.init.constant_(self.fc_t.weight, 0.0)

        self.scale = scale
        if self.scale:
            self.scaler = nn.Parameter(torch.FloatTensor(1, n_out))
            nn.init.constant_(self.scaler, 1.0)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """Perform forward propagation.

        Parameters:
            x: input tensor.
        """
        hidden = F.relu(self.fc1(x))
        if self.scale:
            s = self.scaler * torch.tanh(self.fc_s(hidden))
        else:
            s = torch.tanh(self.fc_s(hidden))
        t = self.fc_t(hidden)
        return s, t

    def randomize_weights(self):
        """Perform random initialization of the trainable parameters."""
        nn.init.normal_(self.fc1.weight, 0, 1. / self.fc1.weight.shape[1])
        nn.init.normal_(self.fc_s.weight, 0, 1. / self.fc_s.weight.shape[1])
        nn.init.normal_(self.fc_t.weight, 0, 1. / self.fc_t.weight.shape[1])
        if self.scale:
            nn.init.uniform_(self.scaler)
