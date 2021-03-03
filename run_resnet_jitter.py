#!/usr/bin/env python3

import torchvision
import torch
from architectures import PlainResNet18
import sys
from torchvision.transforms import Compose, RandomChoice, RandomOrder, ColorJitter, ToTensor, Normalize, Pad
from torchvision.transforms import RandomCrop, RandomHorizontalFlip

def make_data(data_path):
    train_transforms = Compose([
        RandomOrder([
            RandomChoice([
                ColorJitter(brightness=(0.1,1.0)),
                ColorJitter(brightness=(1.0, 10.0))
            ]),
            RandomChoice([
                ColorJitter(contrast=(0.0, 1.0)),
                ColorJitter(contrast=(1.0, 10.0))
            ])
        ]),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        Pad(2),
        RandomCrop(88),
        RandomHorizontalFlip(p=0.5)
    ])
    test_transforms = Compose([
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    _train_data = torchvision.datasets.STL10(data_path, split='train', download=True,
                transform = train_transforms)
    _test_data = torchvision.datasets.STL10(data_path, split='test', download=True,
                transform = test_transforms)
    train_data_loader = torch.utils.data.DataLoader(_train_data,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=4)
    test_data_loader = torch.utils.data.DataLoader(_test_data,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=4)
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
    print('Train Loss:', torch.tensor(losses).float().mean())
    print('Train Accuracy:', good/total*100)

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
    print('Test Loss:', torch.tensor(losses).float().mean())
    print('Test Accuracy:', good/total*100)

def main(save_path, data_path):
    EPOCHS = 200
    net = PlainResNet18().cuda()
    opt = torch.optim.AdamW(net.parameters(), weight_decay=0.02)
    lossf = torch.nn.CrossEntropyLoss()
    train_data, test_data = make_data(data_path)
    for epoch in range(EPOCHS):
        print('Epoch:', epoch)
        train(net, opt, lossf, train_data)
        test(net, lossf, test_data)
        torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    save_path = 'models/plain_resnet18_jitter.pth'
    data_path = 'data/stl10'
    if len(sys.argv) > 1 and sys.argv[1] == 'g':
        save_path = '/content/drive/MyDrive/mgr/models/plain_resnet18_jitter.pth'
        data_path = '/content/drive/MyDrive/mgr/stl10'
    main(save_path, data_path)

