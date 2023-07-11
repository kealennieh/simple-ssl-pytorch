import torch
import torch.nn as nn
from models.backbone import build_backbone


class SimSiam(nn.Module):
    """
    SimSiam: Exploring Simple Siamese Representation Learning
    Link: https://arxiv.org/abs/2011.10566
    Implementation: https://github.com/facebookresearch/simsiam
    """

    def __init__(self, config):
        super().__init__()
        backbone_name = config["backbone"]["name"]
        projection_dim = config.get("projection_dim", 128)
        proj_hidden_dim = config.get("proj_hidden_dim", 1024)
        pred_hidden_dim = config.get("pred_hidden_dim", 1024)

        backbone = build_backbone(backbone_name, config["backbone"])
        feature_size = backbone.get_feature_size()

        projector = Projector(
            feature_size, hidden_dim=proj_hidden_dim, out_dim=projection_dim
        )
        self.predictor = Predictor(
            in_dim=projection_dim, hidden_dim=pred_hidden_dim, out_dim=projection_dim
        )
        self.encoder = nn.Sequential(backbone, projector)

    def forward(self, x):
        x1, x2 = x
        z1, z2 = self.encoder(x1), self.encoder(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)

        return (z1, z2, p1, p2)


class Projector(nn.Module):
    """
    Projection Head for SimSiam
    """

    def __init__(self, in_dim, hidden_dim=2048, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim), nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Predictor(nn.Module):
    """
    Predictor for SimSiam
    """

    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
