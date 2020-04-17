from model.backbone import xception


def build_backbone(backbone, filters, Norm_Layer):
    return xception.AlignedXception(filters, Norm_Layer)
