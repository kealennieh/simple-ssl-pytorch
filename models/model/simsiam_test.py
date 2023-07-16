"""
python ./models/model/simsiam_test.py
"""

import torch
import unittest
from simsiam import SimSiam


class TestSimSiam(unittest.TestCase):
    def test_simsiam(self):
        config = {
            "backbone": {"name": "Resnet50", "pretrained": False},
            "projection_dim": 128,
            "hidden_proj_dim": 1024,
            "hidden_pred_dim": 1024,
        }

        model = SimSiam(config)

        x1 = torch.rand(4, 3, 224, 224)
        x2 = torch.rand(4, 3, 224, 224)

        x = (x1, x2)

        result = model(x)

        z1 = result[0]
        print(z1)


if __name__ == "__main__":
    unittest.main()
