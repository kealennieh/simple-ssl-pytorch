import torch
import torch.nn as nn
import copy
from models.backbone import build_backbone


class BYOL(nn.Module):
    """
    BYOL: Bootstrap your own latent: A new approach to self-supervised Learning
    Link: https://arxiv.org/abs/2006.07733
    Implementation: https://github.com/deepmind/deepmind-research/tree/master/byol
    """

    def __init__(self, config):
        super().__init__()

        backbone_name = config["backbone"]["name"]
        projection_dim = config.get("projection_dim", 256)
        hidden_dim = config.get("hidden_dim", 4096)
        self.tau = config.get("tau", 0.996)  # for EMA update

        backbone = build_backbone(backbone_name, config["backbone"])
        feature_size = backbone.get_feature_size()

        projector = MLP(feature_size, hidden_dim=hidden_dim, out_dim=projection_dim)

        self.online_encoder = nn.Sequential(backbone, projector)
        self.online_predictor = MLP(
            in_dim=projection_dim, hidden_dim=hidden_dim, out_dim=projection_dim
        )
        self.target_encoder = copy.deepcopy(
            self.online_encoder
        )  # target must be a deepcopy of online, since we will use the backbone trained by online

    def forward(self, x):
        x1, x2 = x
        z1_o, z2_o = self.online_encoder(x1), self.online_encoder(x2)
        p1_o, p2_o = self.online_predictor(z1_o), self.online_predictor(z2_o)
        with torch.no_grad():
            self._momentum_update_target_encoder()
            z1_t, z2_t = self.target_encoder(x1), self.target_encoder(x2)

        return (p1_o, p2_o, z1_t, z2_t)

    def _init_target_encoder(self):
        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data.copy_(param_o.data)
            param_t.requires_grad = False

    @torch.no_grad()
    def _momentum_update_target_encoder(self):
        for param_o, param_t in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_t.data = self.tau * param_t.data + (1.0 - self.tau) * param_o.data


class MLP(nn.Module):
    """
    Projection Head and Prediction Head for BYOL
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
