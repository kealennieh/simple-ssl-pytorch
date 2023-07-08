import torch
import torch.nn as nn
import torchvision


class Resnet50(nn.Module):
    def __init__(self, config):
        super().__init__()
        pretrained = config["pretrained"]

        self.backbone = torchvision.models.resnet50(pretrained)
        self.feature_size = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Identity()

    def get_feature_size(self):
        return self.feature_size

    def forward(self, x):
        output = self.backbone(x)

        return output
