import math
import random
import numpy as np


TRANSFORM_FACTORY_DICT = dict()


def register_transform(name, model):
    if name in TRANSFORM_FACTORY_DICT.keys():
        print("%s is already registered" % (name))
        return

    TRANSFORM_FACTORY_DICT[name] = model


def build_transform(name, model_config):
    if name not in TRANSFORM_FACTORY_DICT.keys():
        print("%s not in transform_dict" % (name))
        return

    return TRANSFORM_FACTORY_DICT[name](model_config)


class TransformClass:
    def __init__(self, config):
        self.build(config)

    def build(self, config):
        self.func_list = []
        for val_dict in config:
            transform_func = build_transform(val_dict["name"], val_dict["config"])
            self.func_list.append(transform_func)

    def __call__(self, data):
        for func in self.func_list:
            data = func(data)

        return data


class CopyTwoImageClass:
    def __init__(self, config):
        super().__init__()

    def __call__(self, inputs):
        img = inputs[0]
        outputs = (img, img)
        return outputs


class TransposeClass:
    def __init__(self, config):
        self.transpose_param = config["transpose_param"]

    def __call__(self, inputs):
        if isinstance(inputs, tuple):
            outputs = []
            for val in inputs:
                val = np.transpose(val, self.transpose_param)
                outputs.append(val)
        else:
            outputs = np.transpose(inputs, self.transpose_param)

        return outputs


register_transform("CopyTwoImageClass", CopyTwoImageClass)
register_transform("TransposeClass", TransposeClass)
