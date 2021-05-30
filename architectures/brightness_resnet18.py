import torch
import torch.nn as nn
from torchsummary import summary

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MeanNorm2d(nn.Module):
    def __init__(self, cin):
        super(MeanNorm2d, self).__init__()

    def forward(self, X):   # X is [B,C,D,H,W]
        return X - torch.mean(X, dim=(-3,-2,-1), keepdim=True)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_block=torch.nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_block(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_block(planes)
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


class BResNet18(nn.Module):
    def __init__(self, equivariant=True, layers=[2,2,2,2], num_classes=10):
        super(BResNet18, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        if equivariant:
            self.bn0 = nn.Identity()
            self.norm_block = MeanNorm2d
        if not equivariant:     # invariant
            self.bn0 = nn.InstanceNorm2d(3, affine=False, track_running_stats=False, eps=0)
            self.norm_block = torch.nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_block(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.dropout = torch.nn.Dropout2d(p=0.25)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                self.norm_block(planes)
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample, self.norm_block))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.bn0(x)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.dropout(y)
        y = self.layer3(y)
        y = self.dropout(y)
        y = self.layer4(y)
        y = self.dropout(y)

        y = self.avgpool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return y


if __name__ == '__main__':
    net = BResNet18().cuda()
    inp = (3,80,80)
    print(summary(net, inp))