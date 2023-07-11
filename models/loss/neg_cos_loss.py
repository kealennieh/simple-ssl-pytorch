import torch
import torch.nn as nn
import torch.nn.functional as F


class NegCosLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        z1, z2, p1, p2 = x
        loss = self.neg_cos_similarity(p1, z2) / 2 + self.neg_cos_similarity(p2, z1) / 2

        return loss

    def neg_cos_similarity(self, p, z):
        """
        egative Cosine Similarity
        """
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()
