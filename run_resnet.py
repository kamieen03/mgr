#!/usr/bin/env python3

import torchvision
import torch
from architectures import PlainResNet18
import sys

def make_data(data_path):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        torchvision.transforms.RandomCrop(80),
        torchvision.transforms.RandomHorizontalFlip(p=0.5)
    ])
    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    _train_data = torchvision.datasets.STL10(data_path, split='train', download=True,
                transform = train_transforms)
    _test_data = torchvision.datasets.STL10(data_path, split='test', download=True,
                transform = test_transforms)
    train_data_loader = torch.utils.data.DataLoader(_train_data,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=2)
    test_data_loader = torch.utils.data.DataLoader(_test_data,
                                              batch_size=8,
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
    EPOCHS = 300
    net = PlainResNet18().cuda()
    opt = torch.optim.Adam(net.parameters(), weight_decay=0.001)
    lossf = torch.nn.CrossEntropyLoss()
    train_data, test_data = make_data(data_path)
    for epoch in range(EPOCHS):
        print('Epoch:', epoch)
        train(net, opt, lossf, train_data)
        test(net, lossf, test_data)
        torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    save_path = 'models/plain_resnet18.pth'
    data_path = 'data/stl10'
    if len(sys.argv) > 1 and sys.argv[1] == 'g':
        save_path = '/content/drive/MyDrive/mgr/models/plain_resnet18.pth'
        data_path = '/content/drive/MyDrive/mgr/stl10'
    main(save_path, data_path)

