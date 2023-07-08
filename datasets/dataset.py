import os
import pickle
import numpy as np
from torch.utils.data import Dataset


DATASET_FACTORY_DICT = dict()


def register_dataset(name, model):
    if name in DATASET_FACTORY_DICT.keys():
        print("%s is already registered" % (name))
        return
    DATASET_FACTORY_DICT[name] = model


def build_dataset(name, dataset_config):
    if name not in DATASET_FACTORY_DICT.keys():
        print("%s not in dataset_dict" % (name))
        return

    return DATASET_FACTORY_DICT[name](dataset_config)


class DebugDataset(Dataset):
    def __init__(self, config):
        super().__init__()
        self.length = 300
        self.data = [
            np.random.random([3, 224, 224]).astype(np.float32)
            for _ in range(self.length)
        ]

    def __getitem__(self, index):
        img1 = self.data[index]
        img2 = self.data[index] + 0.0001

        return (img1, img2)

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


register_dataset("DebugDataset", DebugDataset)
