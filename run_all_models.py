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

def train(net, opt, lossf, data):
    losses = []
    good, total = 0, 0
    net.train()
    for X, y in data:
        #torchvision.transforms.functional.to_pil_image(X[0]).show()
        #input()
        X, y = X.cuda(), y.cuda()
        opt.zero_grad()
        out = net(X)
        loss = lossf(out, y)
        losses.append(loss.item())
        loss.backward()
        opt.step()
        good += sum(out.argmax(1) == y)
        total += len(y)
    return torch.tensor(losses).float().mean().item(), (good/total*100).item()

def test(net, lossf, data):
    losses = []
    good, total = 0, 0
    net.eval()
    with torch.no_grad():
        for X, y in data:
            X, y = X.cuda(), y.cuda()
            out = net(X)
            loss = lossf(out, y)
            losses.append(loss.item())
            good += sum(out.argmax(1) == y)
            total += len(y)
    return torch.tensor(losses).float().mean().item(), (good/total*100).item()

def main():
    EPOCHS = 150
    base_model_path = '/content/drive/MyDrive/mgr/models'
    base_runs_path = '/content/drive/MyDrive/mgr/runs'
    data_path = '/content/stl10'
    #base_model_path = 'models'
    #base_runs_path = 'runs'
    #data_path = 'data/stl10'
    for JITTER in [False, True]:
        train_data, test_data = make_data(data_path, JITTER)
        for net, net_name in model_list():
            net = net.cuda()
            opt = torch.optim.AdamW(net.parameters(), weight_decay=0.02)
            lossf = torch.nn.CrossEntropyLoss()
            tr_L, te_L, tr_Acc, te_Acc = [], [], [], []
            for epoch in range(EPOCHS):
                print(net_name, JITTER, epoch)
                train_loss, train_acc = train(net, opt, lossf, train_data)
                test_loss, test_acc = test(net, lossf, test_data)
                torch.save(net.state_dict(), f'{base_model_path}/{net_name}_{JITTER}.pth')

                tr_L.append(train_loss); tr_Acc.append(train_acc)
                te_L.append(test_loss); te_Acc.append(test_acc)
                with open(f'{base_runs_path}/{net_name}_{JITTER}.txt', 'w') as f:
                    f.write(str(tr_L)+'\n')
                    f.write(str(tr_Acc)+'\n')
                    f.write(str(te_L)+'\n')
                    f.write(str(te_Acc)+'\n')

if __name__ == '__main__':
    main()

