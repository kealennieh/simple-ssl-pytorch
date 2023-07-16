"""
python ./models/model/moco_v3_test.py
"""

import torch
import unittest
from moco_v3 import MoCoV3


class TestMoCoV3(unittest.TestCase):
    def test_moco_v3(self):
        config = {
            "backbone": {"name": "Resnet50", "pretrained": False},
            "projection_dim": 256,
            "hidden_dim": 4096,
        }

        model = MoCoV3(config)

        x1 = torch.rand(4, 3, 224, 224)
        x2 = torch.rand(4, 3, 224, 224)

        x = (x1, x2)

        result = model(x)

        z1 = result[0]
        print(z1)


if __name__ == "__main__":
    unittest.main()
