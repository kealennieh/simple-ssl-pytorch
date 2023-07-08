"""
python ./models/model/simclr_test.py
"""

import torch
from simclr import SimCLR


def test_simclr():
    config = {
        "backbone": {"name": "Resnet50", "pretrained": False},
        "projection_dim": 128,
    }

    model = SimCLR(config)

    x1 = torch.rand(4, 3, 224, 224)
    x2 = torch.rand(4, 3, 224, 224)

    x = (x1, x2)

    result = model(x)

    z1 = result[0]
    print(z1)


if __name__ == "__main__":
    test_simclr()
