"""
python ./models/model/moco_test.py
"""

import torch
from moco import MoCo


def test_simclr():
    config = {
        "backbone": {"name": "Resnet50", "pretrained": False},
        "projection_dim": 128,
        "momentum": 0.999,
    }

    model = MoCo(config)

    x_q = torch.rand(4, 3, 224, 224)
    x_k = torch.rand(4, 3, 224, 224)

    x = (x_q, x_k)

    result = model(x)

    z1 = result[0]
    print(z1)


if __name__ == "__main__":
    test_simclr()
