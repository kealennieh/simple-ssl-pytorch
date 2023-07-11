import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanSquaredLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        p1_o, p2_o, z1_t, z2_t = x

        loss = (
            self.mean_squared_error(p1_o, z2_t) / 2
            + self.mean_squared_error(p2_o, z1_t) / 2
        )
        return loss

    def mean_squared_error(self, p, z):
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z.detach()).sum(dim=-1).mean()
