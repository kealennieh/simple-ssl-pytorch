import torch
import torch.nn as nn
from models.backbone import build_backbone


class SimCLRV2(nn.Module):
    """
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
    Link: https://arxiv.org/abs/2002.05709
    Imple
    """

    def __init__(self, config):
        super().__init__()
        backbone_name = config["backbone"]["name"]
        projection_dim = config.get("projection_dim", 128)

        backbone = build_backbone(backbone_name, config["backbone"])
        feature_size = backbone.get_feature_size()
        projector = Projector(
            feature_size, hidden_dim=feature_size, out_dim=projection_dim
        )
        self.encoder = nn.Sequential(backbone, projector)

    def forward(self, x):
        x1, x2 = x
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)

        return (z1, z2)

    @torch.no_grad()
    def eval(self):
        super().eval()
        self.backbone = nn.Sequential(self.backbone, self.projector.layer1)


class Projector(nn.Module):
    """Projector for SimCLR v2"""

    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
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
