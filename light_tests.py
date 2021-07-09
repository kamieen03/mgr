#!/usr/bin/env python3

import torch
import numpy as np
import sys

sys.path.append('architectures')
from architectures.groups import RplusContrast, RplusGamma
from abc import ABC

def make_data(data_path, dataset, jitter):
    test_transforms = ToTensor()
    if dataset == STL10:
        _test_data = STL10(data_path, split='test', download=True, transform = test_transforms)
    else:
        _test_data = dataset(data_path, train=False, download=True, transform = test_transforms)

    test_data_loader = torch.utils.data.DataLoader(_test_data,
                                              batch_size=256,
                                              shuffle=True,
                                              num_workers=2)
    return test_data_loader


def _abstract_test_single(net, net_name, base_dataset, transform, factor):
    losses = []
    good, total = 0, 0
    net.eval()
    lossf = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in base_dataset:
            X, y = X.cuda(), y.cuda()
            X = transform(X.cuda(), factor)
            out = net(X)
            loss = lossf(out, y)
            losses.append(loss.item())
            good += sum(out.argmax(1) == y)
            total += len(y)
    return torch.tensor(losses).float().mean().item(), (good/total*100).item()

def _abstract_test_multi(net, net_name, base_dataset, transform, factors):
    losses, accs = [], []
    for f in factors:
        loss, acc = _abstract_test_single(net, net_name, base_dataset, transform, f)
        losses.append(loss)
        accs.append(acc)
    return losses, accs


class LightTest(ABC):
    transform = None
    factors = []
    name = 'name'

class ContrastTest(LightTest):
    transform = lambda x, a: torch.clip(RplusContrast.transform_tensor(x,a), 0, 1)
    factors = 2**np.linspace(np.log2(0.2), np.log2(5), 11)
    name = 'contrast'

class BrightnessTest(LightTest):
    transform = lambda x, a: torch.clip(x*a, 0, 1)
    factors = 2**np.linspace(np.log2(0.2), np.log2(5), 11)
    name = 'brightness'

class ColorBalanceTest(LightTest):
    t_to_c = {1000: [1.0000, 0.0401, 0.0000],
              2000: [1.0000, 0.2484, 0.0061],
              3000: [1.0000, 0.4589, 0.1483],
              4000: [1.0000, 0.6354, 0.3684],
              5000: [1.0000, 0.7792, 0.6180],
              6000: [1.0000, 0.8952, 0.8666],
              7000: [0.9102, 0.9000, 1.0000],
              8000: [0.7644, 0.8139, 1.0000],
              9000: [0.6693, 0.7541, 1.0000],
              10000: [0.6033, 0.7106, 1.0000],
              11000: [0.5551, 0.6776, 1.0000]}

    def transform(X, t):
        c = ColorBalanceTest.t_to_c[t]
        if len(X.shape) == 3:
            return torch.stack([X[0]*c[0], X[1]*c[1], X[2]*c[2]], dim=0)
        elif len(X.shape) == 4:
            return torch.stack([X[:,0,:,:]*c[0], X[:,1,:,:]*c[1], X[:,2,:,:]*c[2]],
                    dim=1)

    factors = np.arange(1000, 11001, 1000)
    name = 'color_balance'

class GammaTest(LightTest):
    transform = lambda x, a: x**a
    factors = 2**np.linspace(-np.log2(3), np.log2(3), 11)
    name = 'gamma'

def run_light_tests(net, net_name, base_dataset, base_results_path, jitter):
    for Test in [ContrastTest, BrightnessTest, ColorBalanceTest, GammaTest]:
        loss, acc =  _abstract_test_multi(net, net_name, base_dataset,
                                          Test.transform,
                                          Test.factors)
        with open(f'{base_results_path}/{net_name}_{jitter}_{Test.name}.txt', 'w') as f:
            f.write(str(loss)+'\n')
            f.write(str(acc)+'\n')



if __name__ == '__main__':
    from torchvision.datasets import CIFAR10, STL10, CIFAR100
    from torchvision.transforms import ToTensor
    from architectures.models import stl10_model_list, cifar_model_list
    from sys import argv

    _dataset_str = argv[1]
    dataset = {'stl10': STL10, 'cifar10': CIFAR10,
            'cifar100': CIFAR100}[_dataset_str]
    JITTER = {'False': False, 'True': True}[argv[2]]
    base_model_path = f'/content/drive/MyDrive/mgr/models/{_dataset_str}'
    base_results_path = f'/content/drive/MyDrive/mgr/results/{_dataset_str}'
    data_path = f'/content/{_dataset_str}'

    test_data = make_data(data_path, dataset, JITTER)

    if 'cifar' in _dataset_str:
        model_list = cifar_model_list
    else:
        model_list = stl10_model_list
    if '100' in _dataset_str:
        NUM_CLASSES = 100
    else:
        NUM_CLASSES = 10


    for net, net_name in model_list(NUM_CLASSES):
        model_path = f'{base_model_path}/{net_name}_{JITTER}.pth'
        net = net.cuda()
        net.load_state_dict(torch.load(model_path))
        print(net_name)
        run_light_tests(net, net_name, test_data, base_results_path, JITTER)

