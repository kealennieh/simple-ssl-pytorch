import torch
import torch.nn as nn
from models.backbone import build_backbone


class SimCLR(nn.Module):
    """
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
    Link: https://arxiv.org/abs/2002.05709
    Implementation: https://github.com/google-research/simclr
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


class Projector(nn.Module):
    """
    Projector for SimCLR
    """

    def __init__(self, in_dim, hidden_dim=None, out_dim=128):
        super().__init__()

        if hidden_dim is None:
            self.layer1 = nn.Linear(in_dim, out_dim)
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x):
        x = self.layer1(x)
        return x
