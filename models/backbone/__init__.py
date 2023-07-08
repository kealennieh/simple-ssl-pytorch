from .resnet50 import Resnet50


BACKBONE_FACTORY_DICT = dict()


def register_backbone(name, backbone):
    if name in BACKBONE_FACTORY_DICT.keys():
        print("%s is already registered" % (name))
        return

    BACKBONE_FACTORY_DICT[name] = backbone


def build_backbone(name, backbone_config):
    if name not in BACKBONE_FACTORY_DICT.keys():
        print("%s not in backbone_dict" % (name))
        return

    return BACKBONE_FACTORY_DICT[name](backbone_config)


register_backbone("Resnet50", Resnet50)
