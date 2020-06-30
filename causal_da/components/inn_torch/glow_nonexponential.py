import torch
import torch.nn as nn
import torch.nn.functional as F
import inn_torch.layers as layers
from inn_torch.wrappers import SequentialFlow


class GlowNonExponential(nn.Module):
    def __init__(self, depth, dim, n_hidden=20, use_plu=False):
        super().__init__()
        chain = []
        D, d = dim, dim // 2
        chain.append(layers.ActNorm(dim))
        for i in range(depth):
            if use_plu:
                chain.append(layers.InvertiblePLU(dim))
            else:
                chain.append(layers.InvertibleLinear(dim))
            chain.append(
                layers.NonexponentialAffineCouplingLayer(
                    d, NN(d, n_hidden, D - d)))
        self.net = SequentialFlow(chain)

    def forward(self, x):
        return self.net(x)

    def inv(self, x):
        return self.net.inv(x)

    def randomize_weights(self):
        for net in self.net.chain:
            if isinstance(net, layers.AffineCouplingLayer):
                net.net.randomize_weights()
        return self


class NN(nn.Module):
    def __init__(self, n_input, n_hidden, n_out, scale=False):
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
            nn.init.constant_(self.scaler, 0.0)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        if self.scale:
            s = self.scaler * torch.tanh(self.fc_s(hidden))
        else:
            s = torch.tanh(self.fc_s(hidden))
        t = self.fc_t(hidden)
        return s, t

    def randomize_weights(self):
        nn.init.normal_(self.fc1.weight, 0, 1. / self.fc1.weight.shape[1])
        nn.init.normal_(self.fc_s.weight, 0, 1. / self.fc_s.weight.shape[1])
        nn.init.normal_(self.fc_t.weight, 0, 1. / self.fc_t.weight.shape[1])
        if self.scale:
            nn.init.uniform_(self.scaler)
