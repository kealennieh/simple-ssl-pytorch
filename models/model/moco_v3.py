import copy
import torch
import torch.nn as nn
from models.backbone import build_backbone


class MoCoV3(nn.Module):
    """
    MoCo v3: Momentum Contrast v3
    Link: https://arxiv.org/abs/2104.02057
    Implementation: https://github.com/facebookresearch/moco-v3
    """

    def __init__(self, config):
        super().__init__()

        backbone_name = config["backbone"]["name"]
        projection_dim = config.get("projection_dim", 256)
        hidden_dim = config.get("hidden_dim", 2048)
        self.momentum = config.get("momentum", 0.999)

        backbone = build_backbone(backbone_name, config["backbone"])
        feature_size = backbone.get_feature_size()

        projector = Projector(
            feature_size, hidden_dim=hidden_dim, out_dim=projection_dim
        )

        self.encoder_q = nn.Sequential(backbone, projector)
        self.predictor = Predictor(
            in_dim=projection_dim, hidden_dim=hidden_dim, out_dim=projection_dim
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self._init_encoder_k()

    def forward(self, x):
        x1, x2 = x
        q1 = self.predictor(self.encoder_q(x1))
        q2 = self.predictor(self.encoder_q(x2))
        with torch.no_grad():
            self._update_momentum_encoder()
            k1 = self.encoder_k(x1)
            k2 = self.encoder_k(x2)

        return (q1, q2, k1, k2)

    @torch.no_grad()
    def _update_momentum_encoder(self):
        for param_b, param_m in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                1.0 - self.momentum
            )

    @torch.no_grad()
    def _init_encoder_k(self):
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False


class Projector(nn.Module):
    """
    Projector for SimCLR v2, used in MoCo v3 too
    """

    def __init__(self, in_dim, hidden_dim=2048, out_dim=256):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, eps=1e-5, affine=True),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Predictor(nn.Module):
    """
    Projection Head and Prediction Head for BYOL, used in MoCo v3 too
    """

    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
