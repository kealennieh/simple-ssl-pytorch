import torch
import torch.nn as nn
import torch.nn.functional as F


class BarlowTwinsLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_param = config.get("lambda", 0.005)

    def forward(self, x):
        z1, z2 = x
        bz = z1.shape[0]
        c = z1.T @ z2
        c.div_(bz)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_param * off_diag
        return loss

    def off_diagonal(self, x):
        n, m = x.shape
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
