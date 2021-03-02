#!/usr/bin/env python3

import torchvision
import torch
from architectures import PlainResNet18
import sys

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
                                              num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(_test_data,
                                              batch_size=64,
                                              shuffle=True,
                                              num_workers=4)
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
    net = PlainResNet18().cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    i = 1
    for c in [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:       # 0 is grey image, 1 is original 
        for b in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]: # 0 is black image, 1 is original
            train_set, test_set = make_data(data_path, c, b)
            results[(c,b)] = []
            results[(c,b)].append(test(net, train_set))
            results[(c,b)].append(test(net, test_set))
            print(f'[{i}/49]')
            i += 1
    with open(out_path, 'w') as f:
        f.write('contrast brightness train_set_acc test_set_acc')
        for k, v in results.items():
            f.write(f'{key[0]} {key[1]} {value[0]} {value[1]}')


if __name__ == '__main__':
    model_path = 'models/plain_resnet18.pth'
    data_path = 'data/stl10'
    out_path = 'results/plain_resnet18_tests.txt'
    if len(sys.argv) > 1 and sys.argv[1] == 'g':
        base_path = '/content/drive/MyDrive/mgr/'
        model_path = base_path + model_path
        data_path  = base_path + 'stl10'
        out_path   = base_path + out_path
    main(model_path, data_path, out_path)

