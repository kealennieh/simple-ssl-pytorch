"""
python ./datasets/cifar10_dataset_test.py
"""

import unittest
from cifar10_dataset import Cifar10Dataset


class TestCifar10Dataset(unittest.TestCase):
    def test_cifar10_dataset(self):
        config = {
            "data_root": "data",
            "is_train": True,
            "transform": [
                {"name": "CopyTwoImageClass", "config": {}},
                {"name": "TransposeClass", "config": {"transpose_param": [2, 0, 1]}},
            ],
        }

        dataset_loader = Cifar10Dataset(config)

        idx = 3
        for data in dataset_loader:
            print(data)
            if idx > 3:
                break

            idx += 1


if __name__ == "__main__":
    unittest.main()
