from .simclr import SimCLR
from .simclr_v2 import SimCLRV2
from .moco import MoCo


MODEL_FACTORY_DICT = dict()


def register_model(name, model):
    if name in MODEL_FACTORY_DICT.keys():
        print("%s is already registered" % (name))
        return

    MODEL_FACTORY_DICT[name] = model


def build_model(name, model_config):
    if name not in MODEL_FACTORY_DICT.keys():
        print("%s not in model_dict" % (name))
        return

    return MODEL_FACTORY_DICT[name](model_config)


register_model("SimCLR", SimCLR)
register_model("SimCLRV2", SimCLRV2)
register_model("MoCo", MoCo)
