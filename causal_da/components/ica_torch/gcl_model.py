"""
Generalized Contrastive Learning ICA
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Union, List, Optional


class ComponentwiseTransform(nn.Module):
    def __init__(self, modules: Union[List[nn.Module], nn.ModuleList]):
        super().__init__()
        if isinstance(modules, list):
            self.module_list = nn.ModuleList(modules)
        else:
            self.module_list = modules

    def __call__(self, x, aux=None):
        if aux is not None:
            return torch.cat(tuple(self.module_list[d](x[:, (d, )], aux)
                                   for d in range(x.shape[1])),
                             dim=1)
        else:
            return torch.cat(tuple(self.module_list[d](x[:, (d, )])
                                   for d in range(x.shape[1])),
                             dim=1)


class ComponentWiseTransformWithAuxSelection(ComponentwiseTransform):
    def __init__(self,
                 x_dim: int,
                 n_aux: int,
                 hidden_dim: int = 10,
                 n_layer: int = 1):
        super().__init__([
            nn.Sequential(
                nn.Linear(1, hidden_dim), nn.ReLU(),
                *([nn.Linear(hidden_dim, hidden_dim),
                   nn.ReLU()] * n_layer), nn.Linear(hidden_dim, n_aux))
            for _ in range(x_dim)
        ])

    def __call__(self, x: torch.FloatTensor, aux: torch.LongTensor):
        outputs = torch.cat(tuple(self.module_list[d](x[:, (d, )]).unsqueeze(2)
                                  for d in range(x.shape[1])),
                            dim=2).sum(dim=2)
        result = outputs[torch.arange(len(outputs)), aux.flatten()]
        return result


class _LinearWithAux(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def __call__(self, x, aux=None):
        return self.linear(x), aux


class GeneralizedContrastiveICAModel(nn.Module):
    def __init__(
            self,
            network: nn.Module,
            dim: int,
            n_label: int,
            componentwise_transform: Optional[ComponentwiseTransform] = None,
            linear: nn.Module = None):
        """
        Params:
            network: An invertible neural network (of PyTorch).
            linear: A linear layer to be placed at the beginning of the classification layer.
            dim: The dimension of the input (i.e., the dimension of the signal source).
        """
        super().__init__()
        self.network = network
        if componentwise_transform is not None:
            self.componentwise_transform = componentwise_transform
        else:
            self.componentwise_transform = ComponentWiseTransformWithAuxSelection(
                dim, n_label)

        if linear is not None:
            self.linear = _LinearWithAux(linear)
            self.classification_net = nn.Sequential(
                self.linear, self.componentwise_transform)
        else:
            self.linear = None
            self.classification_net = self.componentwise_transform

    def hidden(self, x):
        return self.network(x)

    def classify(self, data, return_hidden=True):
        x, u = data
        hidden = self.hidden(x)
        output = self.classification_net(hidden, u)
        if return_hidden:
            return output, hidden
        else:
            return output

    def extract(self, x):
        return self.hidden(x)

    def inv(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return self.network.inv(x).detach().numpy()

    def forward(self, x):
        return self.hidden(x)

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        return self.forward(x)
