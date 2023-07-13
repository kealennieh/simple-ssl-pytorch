import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCoNCELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        projection_dim = config["projection_dim"]
        self.memory_bank_size = config["memory_bank_size"]
        self.temperature = config["temperature"]

        self.register_buffer(
            "queue", torch.randn(projection_dim, self.memory_bank_size)
        )
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        q, k = x

        self.queue = self.queue.to(q.device)
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        loss = F.cross_entropy(logits, labels)

        self._dequeue_and_enqueue(k)
        return loss

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        bz = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.memory_bank_size % bz == 0
        self.queue[:, ptr : (ptr + bz)] = keys.t()
        ptr = (ptr + bz) % self.memory_bank_size
        self.queue_ptr[0] = ptr


class MoCoV3NCELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temperature = config["temperature"]

    def forward(self, x):
        q1, q2, k1, k2 = x

        loss = self.infonce_loss(q1, k2, self.temperature) + self.infonce_loss(
            q2, k1, self.temperature
        )
        return loss

    def infonce_loss(self, q, k, temperature=0.07):
        """
        InfoNCE loss
        """
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        logits = torch.einsum("nc,mc->nm", [q, k])
        logits /= temperature
        labels = (torch.arange(logits.shape[0], dtype=torch.long)).to(q.device)
        loss = F.cross_entropy(logits, labels)
        return loss
