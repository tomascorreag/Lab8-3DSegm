import torch
import torch.nn as nn


class block(nn.Module):
    def __init__(self, inplanes, planes, Norm_layer, kernel_size=2, stride=2,
                 padding=0, dilation=1, bias=False):
        super(block, self).__init__()

        self.deconv = nn.ConvTranspose3d(inplanes, planes, kernel_size,
                                         stride, padding, bias=bias)
        self.bn1 = Norm_layer(planes, affine=True)
        self.conv = nn.Conv3d(planes * 2, planes, 3, 1, 1, 1, bias=bias)
        self.bn = Norm_layer(planes, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        x = self.deconv(x)
        x = self.relu(self.bn1(x))
        x = self.conv(torch.cat([x, y], dim=1))
        x = self.relu(self.bn(x))
        return x


class Decoder(nn.Module):
    def __init__(self, num_classes, filters, Norm_layer):
        super(Decoder, self).__init__()
        self.conv2 = block(filters[4], filters[3], Norm_layer)
        self.conv3 = block(filters[3], filters[2], Norm_layer)
        self.conv4 = block(filters[2], filters[1], Norm_layer)
        self.conv5 = block(filters[1], filters[0], Norm_layer,
                           kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.final = nn.Conv3d(
            filters[0], num_classes, kernel_size=1, padding=0, bias=False)
        self._init_weight()

    def forward(self, x, x1, x2, x3, x4):
        x3 = self.conv2(x4, x3)
        x2 = self.conv3(x3, x2)
        x1 = self.conv4(x2, x1)
        x = self.conv5(x1, x)
        x = self.final(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    m.bias = nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, filters, Norm_layer):
    return Decoder(num_classes, filters, Norm_layer)
