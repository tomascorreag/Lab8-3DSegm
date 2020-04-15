import torch
import torch.nn as nn
from model.decoder import build_decoder
from model.backbone import build_backbone


class GNet(nn.Module):
    def __init__(self, backbone='xception', num_classes=3):
        super(GNet, self).__init__()
        filters = [32, 64, 128, 256, 512]
        Norm_layer = nn.InstanceNorm3d
        self.backbone = build_backbone(backbone, filters, Norm_layer)
        self.decoder = build_decoder(num_classes, filters, Norm_layer)

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(*x)
        return x


if __name__ == "__main__":
    model = GNet(backbone='resnet')
    model.eval()
    input = torch.rand(1, 1, 48, 48, 48)
    output = model(input)
    print(output.size())
