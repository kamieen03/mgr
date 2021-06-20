from torchinfo import summary

from plain_resnet18 import PlainResNet18
from bspline_resnet18 import BsplineResNet18
from brightness_resnet18 import BResNet18
from groups import SO2, Rplus, RplusContrast, Rshear

def stl10_model_list(num_classes):
    yield PlainResNet18(), 'Plain'
    yield PlainResNet18(inCBW=True), 'Plain+InCBW0'
    yield PlainResNet18(inGBW=True), 'Plain+InGBW0'

    yield BsplineResNet18(N_h=12, h_basis_size=12, group=SO2,
            layer_sizes=[32,64,64,64]), 'RotEq'

    yield BsplineResNet18(N_h=12, h_basis_size=12, group=SO2,
            layer_sizes=[32,64,64,64], inCBW=True), 'RotEq+InCBW0'
    yield BsplineResNet18(N_h=12, h_basis_size=12, group=SO2,
            layer_sizes=[32,64,64,64], inGBW=True), 'RotEq+InGBW0'

    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rplus,
            layer_sizes=[48,64,100,128]), 'ScaleEq'
    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rplus,
            layer_sizes=[48,64,100,128], inCBW=True), 'ScaleEq+InCBW0'
    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rplus,
            layer_sizes=[48,64,100,128], inGBW=True), 'ScaleEq+InGBW0'

    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rshear,
            layer_sizes=[48,64,100,128]), 'SchearEq'
    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rshear,
            layer_sizes=[48,64,100,128], inCBW=True), 'SchearEq+InCBW0'
    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rshear,
            layer_sizes=[48,64,100,128], inGBW=True), 'SchearEq+InGBW0'

    #yield BsplineResNet18(N_h=5, h_basis_size=5, group=RplusContrast,
    #        layer_sizes=[48,64,100,128]), 'ContrastEq'

    yield BResNet18(equivariant=True), 'BrightnessEq'
    yield BResNet18(inB_level=1), 'InB1'
    yield BResNet18(inB_level=2), 'InB2'
    yield BResNet18(inB_level=3), 'InB3'
    return

def cifar_model_list(num_classes=10):
    yield PlainResNet18(layers=[3,3,2], num_classes=num_classes), 'Plain'
    yield PlainResNet18(inCBW=True, layers=[3,3,2], num_classes=num_classes), 'Plain+InCBW0'
    yield PlainResNet18(inGBW=True, layers=[3,3,2], num_classes=num_classes), 'Plain+InGBW0'

    yield BsplineResNet18(N_h=12, h_basis_size=12, group=SO2,
            layers=[3,3,2], layer_sizes=[32,48,64], num_classes=num_classes), 'RotEq'
    yield BsplineResNet18(N_h=12, h_basis_size=12, group=SO2,
            layers=[3,3,2], layer_sizes=[32,48,64], num_classes=num_classes, inCBW=True), 'RotEq+InCBW0'
    yield BsplineResNet18(N_h=12, h_basis_size=12, group=SO2,
            layers=[3,3,2], layer_sizes=[32,48,64], num_classes=num_classes, inGBW=True), 'RotEq+InGBW0'

    yield BsplineResNet18(N_h=3, h_basis_size=3, group=Rplus,
            layers=[3,3,2], layer_sizes=[64,96,128], num_classes=num_classes), 'ScaleEq'
    yield BsplineResNet18(N_h=3, h_basis_size=3, group=Rplus,
            layers=[3,3,2], layer_sizes=[64,96,128], num_classes=num_classes, inCBW=True), 'ScaleEq+InCBW0'
    yield BsplineResNet18(N_h=3, h_basis_size=3, group=Rplus,
            layers=[3,3,2], layer_sizes=[64,96,128], num_classes=num_classes, inGBW=True), 'ScaleEq+InGBW0'

    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rshear,
            layers=[3,3,2], layer_sizes=[48,72,108], num_classes=num_classes), 'SchearEq'
    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rshear,
            layers=[3,3,2], layer_sizes=[48,72,108], num_classes=num_classes, inCBW=True), 'SchearEq+InCBW0'
    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rshear,
            layers=[3,3,2], layer_sizes=[48,72,108], num_classes=num_classes, inGBW=True), 'SchearEq+InGBW0'

    #yield BsplineResNet18(N_h=5, h_basis_size=5, group=RplusContrast,
    #        layer_sizes=[48,64,100,128]), 'ContrastEq'

    yield BResNet18(equivariant=True, layers=[3,3,2], num_classes=num_classes), 'BrightnessEq'
    yield BResNet18(inB_level=1, layers=[3,3,2], num_classes=num_classes), 'InB1'
    yield BResNet18(inB_level=2, layers=[3,3,2], num_classes=num_classes), 'InB2'
    yield BResNet18(inB_level=3, layers=[3,3,2], num_classes=num_classes), 'InB3'
    return

if __name__ == '__main__':
    for net, net_name in cifar_model_list():
        inp = (1,3,32,32)
        summary(net.cuda(), inp)


