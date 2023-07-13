"""
python ./models/model/barlow_twins_test.py
"""

import torch
from barlow_twins import BarlowTwins


def test():
    config = {
        "backbone": {"name": "Resnet50", "pretrained": False},
        "projection_dim": 256,
        "hidden_dim": 4096,
    }

    model = BarlowTwins(config)

    x1 = torch.rand(4, 3, 224, 224)
    x2 = torch.rand(4, 3, 224, 224)

    x = (x1, x2)

    result = model(x)

    z1 = result[0]
    print(z1)


if __name__ == "__main__":
    test()
