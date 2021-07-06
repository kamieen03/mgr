#!/usr/bin/env python3

import torchvision
import torch
from torchvision.transforms import Compose, RandomChoice, RandomOrder, ColorJitter, ToTensor, Normalize, Pad
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
from torchvision.datasets import STL10, CIFAR10, CIFAR100
import sys

sys.path.append('architectures')
from architectures.models import stl10_model_list, cifar_model_list
from light_tests import run_light_tests
from dropbox_api import upload_all

def get_train_transforms(jitter, dataset):
    if dataset == STL10:
        crop_size = 88
    else:
        crop_size = 32
    if not jitter:
        return Compose([
            ToTensor(),
            Pad(2),
            RandomCrop(crop_size),
            RandomHorizontalFlip(p=0.5)
        ])
    else:
        return Compose([
            RandomChoice([
                ColorJitter(brightness=(0.3, 1.0)),
                ColorJitter(contrast=(0.3, 1.0))
            ]),
            ToTensor(),
            Pad(2),
            RandomCrop(crop_size),
            RandomHorizontalFlip(p=0.5)
        ])

def make_data(data_path, dataset, jitter):
    train_transforms = get_train_transforms(jitter, dataset)
    test_transforms = ToTensor()
    if dataset == STL10:
        _train_data = STL10(data_path, split='train', download=True, transform = train_transforms)
        _test_data = STL10(data_path, split='test', download=True, transform = test_transforms)
    else:
        _train_data = dataset(data_path, train=True, download=True, transform = train_transforms)
        _test_data = dataset(data_path, train=False, download=True, transform = test_transforms)

    train_data_loader = torch.utils.data.DataLoader(_train_data,
                                              batch_size=256,
                                              shuffle=True,
                                              num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(_test_data,
                                              batch_size=256,
                                              shuffle=True,
                                              num_workers=2)
    return train_data_loader, test_data_loader

def train(net, opt, lossf, data, net_name):
    losses = []
    good, total = 0, 0
    net.train()
    for X, y in data:
        #torchvision.transforms.functional.to_pil_image(X[0]).show()
        #input()
        X, y = X.cuda(), y.cuda()
        if 'GBW' in net_name:
            X += 1/255
        opt.zero_grad()
        out = net(X)
        loss = lossf(out, y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
        good += sum(out.argmax(1) == y)
        total += len(y)
    return torch.tensor(losses).float().mean().item(), (good/total*100).item()

def test(net, lossf, data, net_name):
    losses = []
    good, total = 0, 0
    net.eval()
    with torch.no_grad():
        for X, y in data:
            X, y = X.cuda(), y.cuda()
            if 'GBW' in net_name:
                X += 1/255
            out = net(X)
            loss = lossf(out, y)
            losses.append(loss.item())
            good += sum(out.argmax(1) == y)
            total += len(y)
    return torch.tensor(losses).float().mean().item(), (good/total*100).item()

def load_run_results(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    tr_L, tr_Acc = lines[0], lines[1]
    te_L, te_Acc = lines[2], lines[3]
    tr_L = [float(c) for c in tr_L[1:-2].split(',')]
    tr_Acc = [float(c) for c in tr_Acc[1:-2].split(',')]
    te_L = [float(c) for c in te_L[1:-2].split(',')]
    te_Acc = [float(c) for c in te_Acc[1:-2].split(',')]
    return tr_L, te_L, tr_Acc, te_Acc

def main(argv):
    if argv[1] == 'True':
        JITTER = True
    elif argv[1] == 'False':
        JITTER = False
    else:
        raise Exception('Wrong JITTER argument')
    if argv[2] == 'stl10':
        DATASET = STL10
        model_list = stl10_model_list
        NUM_CLASSES = 10
    elif argv[2] == 'cifar10':
        DATASET = CIFAR10
        model_list = cifar_model_list
        NUM_CLASSES = 10
    elif argv[2] == 'cifar100':
        DATASET = CIFAR100
        model_list = cifar_model_list
        NUM_CLASSES = 100
    else:
        raise Exception('Wrong dataset!')
    _dataset_str = argv[2]
    if argv[3] == 'local':
        base_model_path = f'models/{_dataset_str}'
        base_runs_path = f'runs/{_dataset_str}'
        base_results_path = f'results/{_dataset_str}'
        data_path = f'data/{_dataset_str}'
    elif argv[3] == 'g':
        base_model_path = f'/content/drive/MyDrive/mgr/models/{_dataset_str}'
        base_runs_path = f'/content/drive/MyDrive/mgr/runs/{_dataset_str}'
        base_results_path = f'/content/drive/MyDrive/mgr/results/{_dataset_str}'
        data_path = f'/content/{_dataset_str}'


    START_MODEL_NAME = None
    START_EPOCH = 0
    start = True
    if len(argv) > 4:
        START_MODEL_NAME = argv[4]
        START_EPOCH = int(argv[5])
        start = False

    EPOCHS = 150

    train_data, test_data = make_data(data_path, DATASET, JITTER)
    lossf = torch.nn.CrossEntropyLoss()
    for net, net_name in model_list(NUM_CLASSES):
        model_path = f'{base_model_path}/{net_name}_{JITTER}.pth'
        runs_path = f'{base_runs_path}/{net_name}_{JITTER}.txt'
        tr_L, te_L, tr_Acc, te_Acc = [], [], [], []
        net = net.cuda()
        if START_MODEL_NAME == net_name:
            net.load_state_dict(torch.load(model_path))
            tr_L, te_L, tr_Acc, te_Acc = load_run_results(runs_path)
            start = True

        if start:
            opt = torch.optim.AdamW(net.parameters(), weight_decay=0.02)
            for epoch in range(START_EPOCH, EPOCHS):
                print(net_name, JITTER, epoch)
                train_loss, train_acc = train(net, opt, lossf, train_data,
                        net_name)
                test_loss, test_acc = test(net, lossf, test_data, net_name)
                torch.save(net.state_dict(), model_path)

                tr_L.append(train_loss); tr_Acc.append(train_acc)
                te_L.append(test_loss); te_Acc.append(test_acc)
                with open(runs_path, 'w') as f:
                    f.write(str(tr_L)+'\n')
                    f.write(str(tr_Acc)+'\n')
                    f.write(str(te_L)+'\n')
                    f.write(str(te_Acc)+'\n')
            START_EPOCH = 0
            run_light_tests(net, net_name, test_data, base_results_path, JITTER)
            if argv[3] == 'local':
                upload_all(net_name, JITTER, _dataset_str)

if __name__ == '__main__':
    main(sys.argv)

