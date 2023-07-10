from .nt_xent_loss import NTXentLoss
from .moco_nce_loss import MoCoNCELoss


LOSS_FACTORY_DICT = dict()


def register_loss(name, loss):
    if name in LOSS_FACTORY_DICT.keys():
        print("%s is already registered" % (name))
        return

    LOSS_FACTORY_DICT[name] = loss


def build_loss(name, loss_config):
    if name not in LOSS_FACTORY_DICT.keys():
        print("%s not in loss_dict" % (name))
        return

    return LOSS_FACTORY_DICT[name](loss_config)


register_loss("NTXentLoss", NTXentLoss)
register_loss("MoCoNCELoss", MoCoNCELoss)
