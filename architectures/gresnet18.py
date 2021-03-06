import torch
import torch.nn as nn
from torchsummary import summary
import numpy as np
import torch.nn.functional as F

def change_contrast(X, factor):    # X is either [B,C,D,H,W] or [B,C,H,W]
    if len(X.shape) == 4:
        dim = (-3,-2,-1)
    elif len(X.shape) == 5:
        dim = (-4,-2,-1)
    else:
        raise Exception('Wrong dimensionality of input tensor')
    mean = torch.mean(X, dim=dim, keepdim=True)
    return X*factor + (1-factor)*mean

class Lift(nn.Module):
    def __init__(self, cs):
        super(Lift, self).__init__()
        self.cs = cs

    def forward(self, X):   # X is [B,C,H,W]
        l = [change_contrast(X, c) for c in self.cs]
        return torch.stack(l, dim=2)

class LiftedBatchNorm2d(nn.Module):
    def __init__(self, cin, cs):
        super(LiftedBatchNorm2d, self).__init__()
        self.norm = nn.BatchNorm2d(cin * len(cs), affine=False)

    def forward(self, X):   # X is [B,C,D,H,W]
        b,c,d,h,w = X.shape
        X = X.reshape(b,c*d,h,w)
        X = self.norm(X)
        X = X.reshape(b,c,d,h,w)
        return X


class LiftedConv(nn.Module):
    def __init__(self, cin, cout, cs, size, stride=1, padding=0, bias=None):
        super(LiftedConv, self).__init__()
        self.kernel = nn.Parameter(torch.randn((cout, cin, size, size)), requires_grad=True)
        self.lift = Lift(cs[::-1])
        self.cs = cs
        self.stride = stride
        self.padding = [padding]*4
        self.bias = bias

    def forward(self, X):   # X is tensor of shape [B,C,D,H,W]
        cout = self.kernel.shape[0]
        lifted_kernel = self.lift(self.kernel)
        k_stack = torch.cat([change_contrast(lifted_kernel, c) for c in self.cs], dim=0)
        kcout, kcin, kd, kh, kw = k_stack.shape
        k_stack = k_stack.view(kcout, kcin*kd, kh, kw)
        ib, icin, id, ih, iw = X.shape
        fX = X.view(ib, icin*id, ih, iw)
        fX = F.pad(fX, self.padding, mode='constant', value=fX.mean())
        out = F.conv2d(fX, k_stack, stride=self.stride, padding=0, bias=self.bias)
        _, _, oh, ow = out.shape
        out = out.view(ib, id, cout, oh, ow)
        out = out.transpose(1,2)
        #out = out.view(ib, cout, id, oh, ow)
        return out / len(self.cs)

def conv3x3(cin, cout, cs, stride=1):
    return LiftedConv(cin, cout, cs, size=3, stride=stride, padding=1, bias=None)

def conv1x1(cin, cout, cs, stride=1):
    return LiftedConv(cin, cout, cs, size=1, stride=stride, padding=0, bias=None)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, cs, activation, stride=1, downsample=None, normalization=LiftedBatchNorm2d):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, cs, stride)
        self.bn1 = normalization(planes, cs)
        self.activation = activation
        self.conv2 = conv3x3(planes, planes, cs)
        self.bn2 = normalization(planes, cs)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class GResNet18(nn.Module):
    def __init__(self, layers=[2,2,2,2], num_classes=10):
        super(GResNet18, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.contrasts = 2 ** np.linspace(-2,2,5)
        
        self.lift = Lift(self.contrasts)
        self.conv1 = LiftedConv(3, self.inplanes, self.contrasts, 7, stride=2, padding=3, bias=None)
        self.bn1 = LiftedBatchNorm2d(self.inplanes, self.contrasts)
        self.activation = nn.Softsign()
        self.pool = nn.AvgPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), count_include_pad=False)

        self.dropout = torch.nn.Dropout3d(p=0.25)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.final_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * len(self.contrasts), num_classes)

        #for m in self.modules():
        #    if isinstance(m, LiftedConv):
        #        nn.init.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')  #TODO
        #    elif isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, self.contrasts, stride),
                LiftedBatchNorm2d(planes, self.contrasts),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, self.contrasts, self.activation, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, self.contrasts, self.activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        y = x - torch.mean(x, dim=(-3,-2,-1), keepdim=True)
        y *= 4.5
        y = self.lift(x)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.activation(y)
        y = self.pool(y)

        y = self.layer1(y)
        y = self.layer2(y)
        y = self.dropout(y)
        y = self.layer3(y)
        y = self.dropout(y)
        y = self.layer4(y)
        #y = self.dropout(y)

        y = self.final_avg_pool(y)
        y = torch.flatten(y, 1)
        y = self.fc(y)

        return x


if __name__ == '__main__':
    net = GResNet18().cuda()
    inp = (3,80,80)
    print(summary(net, inp))

# TODO: inicjalizacja warstw konwolucyjnych
# TODO: sprawdzić stopień ekwiwariancji na poszcególnych odcinkach przetwarzania
# TODO: porownać czasy przetwarzania gresnet18 i resnet18
# TODO: obczaić 2 metody normalizacji - bn3d i bn2d na reshapowanym tensorze
# TODO: sprawdzić możliwe schematy kontrastów:
#   * len(cs) = 3 lub 5 przez całą sieć
#   * len(cs) = 15 przez całą sieć
#   * len(cs) startuje z 32 i maleje w kolejnych blokach
# TODO: jeśli trening bd zbyt wolny, dodać kolejne bloki albo poszerzyć sieć
# TODO: rozważyć dodanie pośredniej warstwy liniowej pod koniec, np. agregującej wszystkie kontrasty
#   na danym kanale w jeden kanał

