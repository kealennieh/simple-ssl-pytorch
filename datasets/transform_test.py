"""
python ./datasets/transform_test.py
"""
import unittest
import numpy as np
from transform import TransformClass


class TestTransform(unittest.TestCase):
    def test_transform(self):
        config = {
            "transform": [
                {"name": "CopyTwoImageClass", "config": {}},
                {"name": "TransposeClass", "config": {"transpose_param": [2, 0, 1]}},
            ]
        }

        one = TransformClass(config["transform"])

        data = np.reshape(np.arange(24), [2, 4, 3])
        target = np.ones(1)

        inputs = (data, target)
        output = one(inputs)

        print(output)
        print(output[0].shape)


if __name__ == "__main__":
    unittest.main()
