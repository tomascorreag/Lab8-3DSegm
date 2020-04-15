from model.backbone import resnet, xception, drn, mobilenet


def build_backbone(backbone, filters, Norm_Layer):
    if backbone == 'resnet':
        return resnet.ResNet101(filters, Norm_Layer)
    elif backbone == 'xception':
        return xception.AlignedXception(filters, Norm_Layer)
    elif backbone == 'drn':
        return drn.drn_d_54(filters)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(filters)
    else:
        raise NotImplementedError
