#!/usr/bin/env python3

import torchvision
import torch
from architectures import PlainResNet18
from architectures import GResNet18
import sys
import numpy as np

def make_data(data_path, c, b):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda img:
            torchvision.transforms.functional.adjust_contrast(img, c)),
        torchvision.transforms.Lambda(lambda img:
            torchvision.transforms.functional.adjust_brightness(img, b)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    _train_data = torchvision.datasets.STL10(data_path, split='train', download=True,
                transform = transforms)
    _test_data = torchvision.datasets.STL10(data_path, split='test', download=True,
                transform = transforms)
    train_data_loader = torch.utils.data.DataLoader(_train_data,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(_test_data,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=2)
    return train_data_loader, test_data_loader

def test(net, data):
    lossf = torch.nn.CrossEntropyLoss()
    good, total = 0, 0
    with torch.no_grad():
        for X, y in data:
            X, y = X.cuda(), y.cuda()
            out = net(X)
            good += sum(out.argmax(1) == y)
            total += len(y)
    return good/total*100

def main(model_path, data_path, out_path):
    results = {}
    net = GResNet18().cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    i = 1
    for c in 10**np.linspace(-1,1,15):      # 0 is grey image, 1 is original 
        for b in 10**np.linspace(-1,1,15): # 0 is black image, 1 is original
            train_set, test_set = make_data(data_path, c, b)
            results[(c,b)] = []
            results[(c,b)].append(test(net, train_set))
            results[(c,b)].append(test(net, test_set))
            print(f'[{i}/225]')
            i += 1
    with open(out_path, 'w') as f:
        f.write('contrast brightness train_set_acc test_set_acc\n')
        for k, v in results.items():
            f.write(f'{k[0]} {k[1]} {v[0]} {v[1]}\n')


if __name__ == '__main__':
    model_path = 'models/gresnet18.pth'
    data_path = 'data/stl10'
    out_path = 'results/gresnet18_tests.txt'
    if len(sys.argv) > 1 and sys.argv[1] == 'g':
        base_path = '/content/drive/MyDrive/mgr/'
        model_path = base_path + model_path
        data_path  = base_path + 'stl10'
        out_path   = base_path + out_path
    main(model_path, data_path, out_path)

