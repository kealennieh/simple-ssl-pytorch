import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
from datasets.transform import TransformClass


class Cifar10Dataset(Dataset):
    def __init__(self, config):
        super().__init__()
        data_root = config["data_root"]
        is_train = config["is_train"]
        raw_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=is_train, download=True
        )

        self.data = raw_dataset.data
        self.targets = raw_dataset.targets

        self.transform = TransformClass(config["transform"])

    def __getitem__(self, index):
        img_data = self.data[index]  # []
        img_target = self.targets[index]  # int

        inputs = (img_data, img_target)
        outputs = self.transform(inputs)

        img1 = outputs[0]  # (3, 32, 32)
        img2 = outputs[1]  # (3, 32, 32)
        return (img1, img2)

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        return self.__class__.__name__
