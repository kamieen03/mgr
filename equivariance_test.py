#!/usr/bin/env python3

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
from architectures import GResNet18, PlainResNet18
import torchvision
import numpy as np

def make_data(data_path):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    _train_data = torchvision.datasets.STL10(data_path, split='train', download=False,
                transform = transforms)
    train_data_loader = torch.utils.data.DataLoader(_train_data,
                                              batch_size=64,
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

def test_conv(net, data):
    alphas = np.exp(np.linspace(np.log(0.1), np.log(10), 11))
    ret = {}
    for alpha in alphas:
        ret[alpha] = None
        for X, _ in tqdm(data):
            X = X.cuda()
            l = net(X)
            gl = change_contrast(l, alpha)

            g = change_contrast(X, alpha)
            lg = net(g)

            err = (lg-gl)/(torch.abs(lg) + torch.abs(gl)+1e-9)*2
            err = torch.mean(err, dim=0)
            if ret[alpha] is None:
                ret[alpha] = torch.zeros(err.shape).cuda()
            ret[alpha] += err
        ret[alpha] /= len(data)
        ret[alpha] = ret[alpha].abs().mean()
        print(alpha, ret[alpha].item())
    return ret


def main():
    net = GResNet18()
    #net.load_state_dict(torch.load('models/gresnet18.pth'))
    net.cuda()
    net.eval()
    data_path = 'data/stl10'
    if len(sys.argv) > 1 and sys.argv[1] == 'g':
        data_path = '/content/drive/MyDrive/mgr/stl10'
    train_data = make_data(data_path)

    with torch.no_grad():
        res = test_conv(net, train_data)
        plt.plot(res.keys(), res.values())
        plt.xscale('log')
        plt.show()


if __name__ == '__main__':
    main()
