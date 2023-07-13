import torch
import torch.nn as nn
from models.backbone import build_backbone


class BarlowTwins(nn.Module):
    """
    Barlow Twins
    Link: https://arxiv.org/abs/2104.02057
    Implementation: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """

    def __init__(self, config):
        super().__init__()

        backbone_name = config["backbone"]["name"]
        projection_dim = config.get("projection_dim", 8192)
        hidden_dim = config.get("hidden_dim", 8192)

        backbone = build_backbone(backbone_name, config["backbone"])
        feature_size = backbone.get_feature_size()

        projector = Projector(feature_size, hidden_dim, projection_dim)
        self.encoder = nn.Sequential(backbone, projector)
        self.bn = nn.BatchNorm1d(projection_dim, affine=False)

    def forward(self, x):
        x1, x2 = x
        z1, z2 = self.encoder(x1), self.encoder(x2)
        d1, d2 = self.bn(z1), self.bn(z2)

        return (d1, d2)


class Projector(nn.Module):
    """Projector for Barlow Twins"""

    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, eps=1e-5, affine=True),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
