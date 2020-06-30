import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg


class InvertiblePLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        w_init = np.linalg.qr(np.random.randn(dim, dim))[0].astype(np.float32)
        np_p, np_l, np_u = scipy.linalg.lu(w_init)
        np_s = np.diag(np_u)

        self.register_buffer('p', torch.from_numpy(np_p))
        # self.register_buffer('sign_s', torch.sign(torch.from_numpy(np_s)))
        # self.log_s = nn.Parameter(torch.log(torch.abs(torch.from_numpy(np_s))))
        self.s = nn.Parameter(torch.from_numpy(np_s))
        self.u = nn.Parameter(torch.triu(torch.from_numpy(np_u), diagonal=1))
        self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
        self.l_mask = nn.Parameter(torch.tril(torch.ones((dim, dim)), diagonal=-1))
        self.register_buffer('eye', torch.eye(dim))

    def forward(self, x):
        l = self.l * self.l_mask + self.eye
        # u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
        u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(1 + self.log_s)
        weight = torch.matmul(self.p, torch.matmul(l, u))
        return torch.matmul(x, weight.t())

    def inv(self, x):
        l = self.l * self.l_mask + self.eye
        u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
        l = torch.inverse(l.double()).float()
        u = torch.inverse(u.double()).float()
        weight = torch.matmul(u, torch.matmul(l, self.p.inverse()))
        return torch.matmul(x, weight.t())


class Inver(nn.Module):
    def __init__(self, dim, init='identity'):
        super().__init__()
        self.init = init
        if self.init == 'randn':
            w_init = np.linalg.qr(np.random.randn(dim, dim))[0].astype(np.float32)
        elif self.init == 'identity':
            w_init = np.linalg.qr(np.eye(dim))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))

    def forward(self, x):
        return torch.matmul(x, self.weight.t())

    def inv(self, x):
        return torch.matmul(x, torch.inverse(self.weight.double()).float())
