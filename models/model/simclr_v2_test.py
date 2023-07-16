"""
python ./models/model/simclr_v2_test.py
"""

import torch
import unittest
from simclr_v2 import SimCLRV2


class TestSimCLRV2(unittest.TestCase):
    def test_simclr(self):
        config = {
            "backbone": {"name": "Resnet50", "pretrained": False},
            "projection_dim": 128,
        }

        model = SimCLRV2(config)

        x1 = torch.rand(4, 3, 224, 224)
        x2 = torch.rand(4, 3, 224, 224)

        x = (x1, x2)

        result = model(x)

        z1 = result[0]
        print(z1)


if __name__ == "__main__":
    unittest.main()
