#!/usr/bin/env python3

import torchvision
import torch
from architectures import PlainResNet18

def make_data():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    _train_data = torchvision.datasets.STL10('data/stl10', split='train', download=True,
                transform = transforms)
    _test_data = torchvision.datasets.STL10('data/stl10', split='test', download=True,
                transform = transforms)
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
    torch.save(net.state_dict(), 'models/plain_resnet18.pth')

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

def main():
    EPOCHS = 30
    net = PlainResNet18().cuda()
    opt = torch.optim.Adam(net.parameters(), weight_decay=0.001)
    lossf = torch.nn.CrossEntropyLoss()
    train_data, test_data = make_data()
    for epoch in range(EPOCHS):
        print('Epoch:', epoch)
        train(net, opt, lossf, train_data)
        test(net, lossf, test_data)

if __name__ == '__main__':
    main()

