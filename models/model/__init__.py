from .simclr import SimCLR
from .simclr_v2 import SimCLRV2
from .moco import MoCo
from .moco_v2 import MoCoV2
from .moco_v3 import MoCoV3
from .simsiam import SimSiam
from .byol import BYOL
from .barlow_twins import BarlowTwins


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
register_model("MoCoV2", MoCoV2)
register_model("MoCoV3", MoCoV3)
register_model("SimSiam", SimSiam)
register_model("BYOL", BYOL)
register_model("BarlowTwins", BarlowTwins)
