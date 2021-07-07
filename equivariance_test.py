#!/usr/bin/env python3

import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
from sys import argv
import json

from torchvision.transforms.functional import rotate as torch_rotate
from torchvision.transforms.functional import resize as torch_resize
from torchvision.transforms.functional import center_crop as torch_center_crop
from torchvision.transforms.functional import adjust_contrast as torch_adjust_contrast
from torchvision.transforms.functional import affine as torch_affine
import PIL

sys.path.append('architectures')
from architectures import GResNet18, PlainResNet18, BResNet18, BsplineResNet18
from architectures.groups import SO2, Rplus, RplusContrast, Rshear, RplusGamma
from architectures.bspline_kernel import Lift, LiftedConv, Projection
from architectures.models import cifar_model_list
from architectures.norms import LogNorm

def update_dict(dict, net):
    norm = torch.linalg.norm
    dict['conv1'].append(norm(net.conv1.weight.grad)/np.prod(
        net.conv1.weight.shape))
    dict['layer1.0.conv1'].append(norm(net.layer1[0].conv1.weight.grad)/np.prod(
        net.layer1[0].conv1.weight.shape))
    dict['layer1.0.conv2'].append(norm(net.layer1[0].conv2.weight.grad)/np.prod(
        net.layer1[0].conv2.weight.shape))
    dict['layer1.1.conv1'].append(norm(net.layer1[1].conv1.weight.grad)/np.prod(
        net.layer1[1].conv1.weight.shape))
    dict['layer1.1.conv2'].append(norm(net.layer1[1].conv2.weight.grad)/np.prod(
        net.layer1[1].conv2.weight.shape))

    dict['layer2.0.conv1'].append(norm(net.layer2[0].conv1.weight.grad)/np.prod(
        net.layer2[0].conv1.weight.shape))
    dict['layer2.0.conv2'].append(norm(net.layer2[0].conv2.weight.grad)/np.prod(
        net.layer2[0].conv2.weight.shape))
    dict['layer2.1.conv1'].append(norm(net.layer2[1].conv1.weight.grad)/np.prod(
        net.layer2[1].conv1.weight.shape))
    dict['layer2.1.conv2'].append(norm(net.layer2[1].conv2.weight.grad)/np.prod(
        net.layer2[1].conv2.weight.shape))

    dict['layer3.0.conv1'].append(norm(net.layer3[0].conv1.weight.grad)/np.prod(
        net.layer3[0].conv1.weight.shape))
    dict['layer3.0.conv2'].append(norm(net.layer3[0].conv2.weight.grad)/np.prod(
        net.layer3[0].conv2.weight.shape))
    dict['layer3.1.conv1'].append(norm(net.layer3[1].conv1.weight.grad)/np.prod(
        net.layer3[1].conv1.weight.shape))
    dict['layer3.1.conv2'].append(norm(net.layer3[1].conv2.weight.grad)/np.prod(
        net.layer3[1].conv2.weight.shape))


def test_bresnet():
    bresnet = BResNet18(equivariant=True, layers=[2,2,2], num_classes=10)
    bresnet.load_state_dict(torch.load('bresnet.pth'))
    optb = torch.optim.AdamW(bresnet.parameters(), weight_decay=0.02)

    plainnet = PlainResNet18(layers=[2,2,2], num_classes=10)
    plainnet.load_state_dict(torch.load('plainnet.pth'))
    optp = torch.optim.AdamW(plainnet.parameters(), weight_decay=0.02)

    data = make_data('data/cifar10')
    lossf = torch.nn.CrossEntropyLoss()

    plain_grads = {'conv1': [],
                   'layer1.0.conv1': [],
                   'layer1.0.conv2': [],
                   'layer1.1.conv1': [],
                   'layer1.1.conv2': [],
                   'layer2.0.conv1': [],
                   'layer2.0.conv2': [],
                   'layer2.1.conv1': [],
                   'layer2.1.conv2': [],
                   'layer3.0.conv1': [],
                   'layer3.0.conv2': [],
                   'layer3.1.conv1': [],
                   'layer3.1.conv2': []}
    b_grads = {k: [] for k in plain_grads}

    for i, (X, y) in enumerate(data):
        optp.zero_grad()
        outp = plainnet(X)
        lossp = lossf(outp, y)
        lossp.backward()
        update_dict(plain_grads, plainnet)

        optb.zero_grad()
        outb = bresnet(X)
        lossb = lossf(outb, y)
        lossb.backward()
        update_dict(b_grads, bresnet)
        print(f'up {i}')
        if i == 100: break
    for k in plain_grads:
        plain_grads[k] = float(np.mean(plain_grads[k]))
        b_grads[k] = float(np.mean(b_grads[k]))
    print(json.dumps({'Plain': plain_grads, 'Bresnet': b_grads}, indent=4))



def make_data(data_path):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    _train_data = torchvision.datasets.CIFAR10(data_path, train=False, download=False,
                transform = transforms)
    train_data_loader = torch.utils.data.DataLoader(_train_data,
                                              batch_size=100,
                                              shuffle=True,
                                              num_workers=2)
    return train_data_loader

def change_contrast(X, factor):
    if len(X.shape) == 4:
        dim = (-3,-2,-1)
    elif len(X.shape) == 5:
        dim = (-4,-2,-1)
    else:
        raise Exception('Wrong dimensionality of input tensor')
    mean = torch.mean(X, dim=dim, keepdim=True)
    return X*factor + (1-factor)*mean

    if len(X.shape) == 4:
        return torchvision.transforms.functional.rotate(X, factor)
    else:
        return torch.stack([torchvision.transforms.functional.rotate(X[:,:,i,:,:], factor)
                for i in range(X.shape[2])], dim=2)

def change_brightness(X, factor):
    return factor * X

def change_gamma(X, factor):
    return torch.sign(X) * X.abs() ** factor

def rotate(X, theta):
    angle = int(theta)
    if len(X.shape) == 4:
        return torch_rotate(X, angle, resample=PIL.Image.BILINEAR,
                expand=True)
    elif len(X.shape) == 5:
        I = np.arange(X.shape[2])
        out = torch.stack([torch_rotate(X[:,:,i,:,:],
                    angle, resample=PIL.Image.BILINEAR, expand=True)
                    for i in I], dim=2)
        return out
    else:
        raise Exception("Wrong X shape")

def scale(X, scale):
    h, w = X.shape[-2:]
    if len(X.shape) == 4:
        return torch_affine(X, 0, (0,0), scale, (0,0), resample=PIL.Image.BILINEAR)
    elif len(X.shape) == 5:
        I = np.arange(X.shape[2])
        out = torch.stack([Rplus.transform_tensor(X[:,:,i,:,:], scale)
                    for i in I], dim=2)
        return out
    else:
        raise Exception("Wrong X shape")

def shear(X, shear):
    if len(X.shape) == 4:
        shear_angle = np.arctan(shear) * 180/np.pi
        return torch_affine(X, angle=0, translate=[0,0], scale=1,
                shear=shear_angle)
    elif len(X.shape) == 5:
        I = np.arange(X.shape[2])
        out = torch.stack([Rshear.transform_tensor(X[:,:,i,:,:], shear)
                    for i in I], dim=2)
        return out
    else:
        raise Exception("Wrong X shape")

def get_transform(t):
    return {'contrast': (change_contrast, 2**np.linspace(np.log2(0.2), np.log2(5), 11)),
            'brightness': (change_brightness, 2**np.linspace(np.log2(0.2), np.log2(5), 11)),
            'gamma': (change_gamma, 2**np.linspace(-np.log2(3), np.log2(3), 11)),
            'rotate': (rotate, np.arange(0.0, 360.0, 30.0)),
            'scale': (scale, (2**0.5)**np.arange(-4,5,1)),
            'shear': (shear, np.linspace(-1.0,1.0,11))}[t]

def test_conv(net, data, net_name, transform_name):
    transform, factors = get_transform(transform_name)

    ret = {}
    ret_abs = {}
    for f in tqdm(factors):
        for idx, (X, _) in enumerate(data):
            X = X.cuda()

            g = transform(X, f)
            lg = net(g)


            l = net(X)
            gl = transform(l, f)
    #        for i in range(12):
    #            plt.subplot(2,12,i+1)
    #            plt.imshow(np.tanh(lg[0,0,i].cpu().numpy()))
    #            plt.subplot(2,12,i+1+12)
    #            plt.imshow(np.tanh(gl[0,0,i].cpu().numpy()))
    #        plt.show()

    #        print(f)
    #        print(lg.mean(dim=(0,1,3,4)))
    #        print(gl.mean(dim=(0,1,3,4)))
            if transform_name == 'rotate':
                if lg.shape[-1] > gl.shape[-1]:
                    lg = torch_center_crop(lg, gl.shape[-2:])
                else:
                    gl = torch_center_crop(gl, lg.shape[-2:])
            if net_name  == 'RotEq':
                gl = torch.roll(gl, int(f//30), dims=2)
            err = (lg-gl)/(torch.abs(lg) + torch.abs(gl)+1e-9)*2
            ret[f] = err.mean().abs().item()
            ret_abs[f] = err.abs().mean().item()
            break

    return {'err': ret, 'err_abs': ret_abs}


def main():
    model_name = argv[1]
    transform_name = argv[2]
    jitter = argv[3]
    STAGE = argv[4]
    path = f'{model_name}_{transform_name}_{jitter}_{STAGE}.json'

    for net, net_name in cifar_model_list():
        if net_name == model_name:
            model = net
            break
    model.load_state_dict(torch.load(f'../wyniki/models/cifar10/{model_name}_{jitter}.pth'))
    model.cuda()
    model.eval()
    data_path = 'data/cifar10'
    if len(sys.argv) > 1 and sys.argv[1] == 'g':
        data_path = '/content/drive/MyDrive/mgr/stl10'
    data = make_data(data_path)

    with torch.no_grad():
        err = test_conv(model, data, model_name, transform_name)
        with open(f'../wyniki/equivariance/{path}', 'w') as f:
            json.dump(err, f, indent = 4)


if __name__ == '__main__':
    #main()
    test_bresnet()
