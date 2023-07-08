"""
python resnet50_test.py
"""

import torch
from resnet50 import Resnet50


def test_resnet50():
    config = {"pretrained": True}

    model = Resnet50(config)

    x = torch.ones(4, 3, 224, 224)
    result = model(x)

    print(result)


if __name__ == "__main__":
    test_resnet50()
