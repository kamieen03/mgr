from torchinfo import summary

from plain_resnet18 import PlainResNet18
from bspline_resnet18 import BsplineResNet18
from brightness_resnet18 import BResNet18
from groups import SO2, Rplus, RplusContrast, Rshear

def model_list():
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

if __name__ == '__main__':
    for net in model_list():
        inp = (1,3,80,80)
        summary(net.cuda(), inp)


