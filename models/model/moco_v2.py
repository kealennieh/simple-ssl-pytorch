import copy
import torch
import torch.nn as nn
from models.backbone import build_backbone


class MoCoV2(nn.Module):
    """
    MoCo v2: Momentum Contrast v2
    Link: https://arxiv.org/abs/2003.04297
    Implementation: https://github.com/facebookresearch/moco
    """

    def __init__(self, config):
        super().__init__()
        backbone_name = config["backbone"]["name"]
        projection_dim = config.get("projection_dim", 128)
        self.momentum = config.get("momentum", 0.999)

        backbone = build_backbone(backbone_name, config["backbone"])
        feature_size = backbone.get_feature_size()

        projector = Projector(feature_size, feature_size, projection_dim)
        self.encoder_q = nn.Sequential(backbone, projector)

        self.encoder_k = copy.deepcopy(self.encoder_q)
        self._init_encoder_k()

    def forward(self, x):
        x_q, x_k = x
        q = self.encoder_q(x_q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._momentum_update_encoder_k()
            x_k, idx_unshuffle = self._batch_shuffle_single_gpu(x_k)
            k = self.encoder_k(x_k)
            k = nn.functional.normalize(k, dim=1)
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        return (q, k)

    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def _momentum_update_encoder_k(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        idx_shuffle = torch.randperm(x.shape[0])
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        return x[idx_unshuffle]


class Projector(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=128):
        super().__init__()
        if hidden_dim == None:
            hidden_dim = in_dim
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        return x
