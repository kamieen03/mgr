import torch
import torch.nn as nn
from torchinfo import summary
from norms import LogNorm

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class PlainResNet18(nn.Module):
    def __init__(self, inCBW=False, inGBW=False, layers=[2,2,2,2], num_classes=10):
        super(PlainResNet18, self).__init__()
        self.inplanes = 80

        if inCBW:
            self.norm0 = nn.InstanceNorm2d(3, affine=False, track_running_stats=False, eps=0)
        elif inGBW:
            self.norm0 = LogNorm()
        else:
            self.norm0 = nn.Identity()

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dropout = torch.nn.Dropout2d(p=0.5)
        self.layer1 = self._make_layer(80, layers[0])
        self.layer2 = self._make_layer(160, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = None
        if len(layers)  > 3:
            self.layer4 = self._make_layer(256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.norm0(x)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.dropout(y)
        y = self.layer2(y)
        y = self.dropout(y)
        y = self.layer3(y)
        if self.layer4 is not None:
            y = self.dropout(y)
            y = self.layer4(y)

        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y


if __name__ == '__main__':
    net = PlainResNet18(inGBW=False, layers=[3,3,2]).cuda()
    inp = (1,3,32,32)
    print(summary(net, inp))
