import os
import yaml
import json
import torch
import pickle
import random
import numpy as np


def set_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_yaml_config(yaml_file):
    with open(yaml_file, "r") as fin:
        val_dict = yaml.safe_load(fin)

    return val_dict


def parse_json_file(json_path):
    with open(json_path, "r") as fin:
        content_list = json.load(fin)

    return content_list


class AverageMeter(object):
    """
    compute and store the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avg_values = list()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.avg_values.append(self.avg)
