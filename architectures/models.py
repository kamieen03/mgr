from torchinfo import summary

from plain_resnet18 import PlainResNet18
from bspline_resnet18 import BsplineResNet18
from brightness_resnet18 import BResNet18
from groups import SO2, Rplus, RplusContrast, Rshear

def model_list():
    yield PlainResNet18()
    yield PlainResNet18(inCBW=True)
    yield PlainResNet18(inGBW=True)

    yield BsplineResNet18(N_h=12, h_basis_size=12, group=SO2,
            layer_sizes=[32,64,64,64])
    yield BsplineResNet18(N_h=12, h_basis_size=12, group=SO2,
            layer_sizes=[32,64,64,64], inCBW=True)
    yield BsplineResNet18(N_h=12, h_basis_size=12, group=SO2,
            layer_sizes=[32,64,64,64], inGBW=True)

    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rplus,
            layer_sizes=[48,64,100,128])
    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rplus,
            layer_sizes=[48,64,100,128], inCBW=True)
    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rplus,
            layer_sizes=[48,64,100,128], inGBW=True)

    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rshear,
            layer_sizes=[48,64,100,128])
    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rshear,
            layer_sizes=[48,64,100,128], inCBW=True)
    yield BsplineResNet18(N_h=5, h_basis_size=5, group=Rshear,
            layer_sizes=[48,64,100,128], inGBW=True)

    yield BsplineResNet18(N_h=5, h_basis_size=5, group=RplusContrast,
            layer_sizes=[48,64,100,128])

    yield BResNet18(equivariant=True)
    yield BResNet18(inB_level=1)
    yield BResNet18(inB_level=2)
    yield BResNet18(inB_level=3)
    return

if __name__ == '__main__':
    for net in model_list():
        inp = (1,3,80,80)
        summary(net.cuda(), inp)


