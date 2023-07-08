import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temperature = config["temperature"]

    def forward(self, pred):
        """ """
        z1, z2 = pred
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
        )
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)
        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]
        negatives = similarity_matrix[~diag].view(2 * N, -1)
        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature
        labels = torch.zeros(2 * N, device=device, dtype=torch.int64)
        loss = F.cross_entropy(logits, labels, reduction="sum") / (2 * N)

        return loss
