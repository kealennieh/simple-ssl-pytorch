"""
python ./models/model/moco_v2_test.py
"""

import torch
from models.model.moco_v2 import MoCoV2


def test():
    config = {
        "backbone": {"name": "Resnet50", "pretrained": False},
        "projection_dim": 128,
        "momentum": 0.999,
    }

    model = MoCoV2(config)

    x_q = torch.rand(4, 3, 224, 224)
    x_k = torch.rand(4, 3, 224, 224)

    x = (x_q, x_k)

    result = model(x)

    z1 = result[0]
    print(z1)


if __name__ == "__main__":
    test()
