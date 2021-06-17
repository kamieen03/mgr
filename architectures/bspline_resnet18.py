import torch
import torch.nn as nn
from torchinfo import summary
from bspline_kernel import Lift, LiftedConv, Projection
from groups import SO2, Rplus, RplusContrast, Rshear
from norms import LogNorm


def conv3x3(in_planes, out_planes, N_h, h_basis_size, group, stride=1):
    return LiftedConv(in_planes, out_planes, 3, N_h, h_basis_size,
            group, stride=stride, padding=1)

def conv1x1(in_planes, out_planes, N_h, h_basis_size, group, stride=1):
    return LiftedConv(in_planes, out_planes, 1, N_h, h_basis_size,
            group, stride=stride)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, N_h, h_basis_size, group, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, N_h, h_basis_size, group, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, N_h, h_basis_size, group)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BsplineResNet18(nn.Module):
    def __init__(self, N_h, h_basis_size, group,
            inCBW=False, inGBW=False,
            layer_sizes=[32,64,64,64], layers=[2,2,2,2], num_classes=10):
        super(BsplineResNet18, self).__init__()
        self.inplanes = layer_sizes[0]

        if inCBW:
            self.norm0 = nn.InstanceNorm2d(3, affine=False, track_running_stats=False, eps=0)
        elif inGBW:
            self.norm0 = LogNorm()
        else:
            self.norm0 = nn.Identity()
        self.lift = Lift(3, self.inplanes, 7, N_h, group, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        self.dropout = torch.nn.Dropout3d(p=0.25)
        self.layer1 = self._make_layer(layer_sizes[0], layers[0], N_h, h_basis_size, group)
        self.layer2 = self._make_layer(layer_sizes[1], layers[1], N_h, h_basis_size, group, stride=2)
        self.layer3 = self._make_layer(layer_sizes[2], layers[2], N_h, h_basis_size, group, stride=2)
        self.layer4 = self._make_layer(layer_sizes[3], layers[3], N_h, h_basis_size, group, stride=2)

        self.projection = Projection()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layer_sizes[3], num_classes)

       # for m in self.modules():
       #     if isinstance(m, nn.Conv2d):
       #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
       #     elif isinstance(m, nn.BatchNorm2d):
       #         nn.init.constant_(m.weight, 1)
       #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, N_h, h_basis_size, group, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, N_h, h_basis_size, group, stride),
                nn.BatchNorm3d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, N_h, h_basis_size, group, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, N_h, h_basis_size, group))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.norm0(x)
        y = self.lift(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.dropout(y)
        y = self.layer3(y)
        y = self.dropout(y)
        y = self.layer4(y)
        #y = self.dropout(y)

        y = self.projection(y)
        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return x


if __name__ == '__main__':
    net = BsplineResNet18(N_h=5, h_basis_size=6, group=SO2,
            layer_sizes=[32,64,64,128]).cuda()
    inp = (1,3,80,80)
    print(summary(net, inp))

