#!/usr/bin/env python3

import torchvision
import torch
from torchvision.transforms import Compose, RandomChoice, RandomOrder, ColorJitter, ToTensor, Normalize, Pad
from torchvision.transforms import RandomCrop, RandomHorizontalFlip
import sys

sys.path.append('architectures')
from architectures.models import model_list


non_jitter_train_transforms = Compose([
    ToTensor(),
    Pad(2),
    RandomCrop(88),
    RandomHorizontalFlip(p=0.5)
])

jitter_train_transforms = Compose([
    RandomOrder([
        RandomChoice([
            ColorJitter(brightness=(0.2, 1.0)),
            ColorJitter(brightness=(1.0, 5.0))
        ]),
        RandomChoice([
            ColorJitter(contrast=(0.2, 1.0)),
            ColorJitter(contrast=(1.0, 5.0))
        ])
    ]),
    ToTensor(),
    Pad(2),
    RandomCrop(88),
    RandomHorizontalFlip(p=0.5)
])

def make_data(data_path, jitter):
    if jitter:
        train_transforms = jitter_train_transforms
    else:
        train_transforms = non_jitter_train_transforms
    test_transforms = Compose([
        ToTensor(),
    ])
    _train_data = torchvision.datasets.STL10(data_path, split='train', download=True,
                transform = train_transforms)
    _test_data = torchvision.datasets.STL10(data_path, split='test', download=True,
                transform = test_transforms)
    train_data_loader = torch.utils.data.DataLoader(_train_data,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(_test_data,
                                              batch_size=128,
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
    START_MODEL_NAME = None
    start = True
    if len(argv) > 2:
        START_MODEL_NAME = argv[2]
        START_EPOCH = int(argv[3])
        start = False

    EPOCHS = 150
    base_model_path = '/content/drive/MyDrive/mgr/models'
    base_runs_path = '/content/drive/MyDrive/mgr/runs'
    data_path = '/content/stl10'
    #base_model_path = 'models'
    #base_runs_path = 'runs'
    #data_path = 'data/stl10'
    train_data, test_data = make_data(data_path, JITTER)
    lossf = torch.nn.CrossEntropyLoss()
    for net, net_name in model_list():
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
                with open(f'{base_runs_path}/{net_name}_{JITTER}.txt', 'w') as f:
                    f.write(str(tr_L)+'\n')
                    f.write(str(tr_Acc)+'\n')
                    f.write(str(te_L)+'\n')
                    f.write(str(te_Acc)+'\n')
            START_EPOCH = 0

if __name__ == '__main__':
    main(sys.argv)

