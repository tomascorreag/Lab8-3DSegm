import torch
import torch.nn as nn


class SeparableConv3d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, padding=1,
                 stride=1, dilation=1, bias=False):
        super(SeparableConv3d, self).__init__()

        self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size, stride,
                               padding, dilation, groups=inplanes, bias=bias)
        self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, inplanes, planes, Norm_layer, reps, stride=1,
                 dilation=1, start_with_relu=True):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv3d(inplanes, planes, 1, stride=stride,
                                  bias=False)
            self.skipbn = Norm_layer(planes, affine=True)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if stride != 1:
            rep.append(self.relu)
            rep.append(SeparableConv3d(inplanes, planes, 3, 1, stride))
            rep.append(Norm_layer(planes, affine=True))
            filters = planes
        else:
            rep.append(self.relu)
            rep.append(SeparableConv3d(inplanes, planes, 3, 1, 1, dilation))
            rep.append(Norm_layer(planes, affine=True))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv3d(filters, filters, 3, 1, 1, dilation))
            rep.append(Norm_layer(filters, affine=True))

        if not start_with_relu:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = x + skip
        return x


class AlignedXception(nn.Module):
    """
    Modified Alighed Xception
    """
    def __init__(self, filters, Norm_layer):
        super(AlignedXception, self).__init__()
        self.conv1 = nn.Conv3d(1, filters[0], 3, padding=1, bias=False)
        self.bn1 = Norm_layer(filters[0], affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(filters[0], filters[0], 3, padding=1,
                               bias=False)
        self.bn2 = Norm_layer(filters[0], affine=True)

        self.block0 = Block(filters[0], filters[1], Norm_layer, reps=2,
                            stride=[2, 2, 1], start_with_relu=False)
        self.block1 = Block(filters[1], filters[2], Norm_layer, reps=2,
                            stride=2)
        self.block2 = Block(filters[2], filters[3], Norm_layer, reps=3,
                            stride=2)
        self.block3 = Block(filters[3], filters[4], Norm_layer, reps=3,
                            stride=2)

        # Init weights
        self._init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x0 = self.block0(x)
        x1 = self.block1(x0)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        return x, x0, x1, x2, x3

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, a=1e-2)
                if m.bias is not None:
                    m.bias = nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    filters = [8, 16, 32, 64, 128, 256]
    model = AlignedXception(filters=filters, Norm_layer=nn.InstanceNorm3d)
    input = torch.rand(1, 3, 48, 48)
    output, low_level_feat = model(input)
    print(output.size())
    print(low_level_feat.size())
