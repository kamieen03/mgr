#!/usr/bin/env python3

import torch
import numpy as np
import sys

sys.path.append('architectures')
from architectures.groups import RplusContrast, RplusGamma
from abc import ABC


def _abstract_test_single(net, net_name, base_dataset, transform, factor):
    losses = []
    good, total = 0, 0
    net.eval()
    lossf = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in base_dataset:
            X, y = X.cuda(), y.cuda()
            if 'GBW' in net_name:
                X += 1/255
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
    transform = RplusContrast.transform_tensor
    factors = 2**np.linspace(np.log2(0.2), np.log2(5), 11)
    name = 'contrast'

class BrightnessTest(LightTest):
    transform = lambda x, a: x*a
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
    transform = RplusGamma.transform_tensor
    factors = 2**np.linspace(-np.log2(3), np.log2(3), 11)
    name = 'gamma'

def run_light_tests(net, net_name, base_dataset, base_results_path, jitter):
    for Test in [ContrastTest, BrightnessTest, ColorBalanceTest, GammaTest]:
        print(Test)
        loss, acc =  _abstract_test_multi(net, net_name, base_dataset,
                                          Test.transform,
                                          Test.factors)
        with open(f'{base_results_path}/{net_name}_{jitter}_{Test.name}.txt', 'w') as f:
            f.write(str(loss)+'\n')
            f.write(str(acc)+'\n')



if __name__ == '__main__':
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(3*32*32,10)
            ).cuda()
    data = CIFAR10('data/cifar10', train=False, download=True, transform = ToTensor())
    data_loader = torch.utils.data.DataLoader(data,
                                              batch_size=16,
                                              shuffle=True,
                                              num_workers=2)
    run_light_tests(net, 'name', data_loader, 'results', False)


