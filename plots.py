from matplotlib import pyplot as plt
import numpy as np
import os
import torch
from torch import nn
import json

class LogNorm(nn.Module):
    def __init__(self):
        super(LogNorm, self).__init__()
        self.inorm = nn.InstanceNorm2d(3, affine=False, track_running_stats=False, eps=0)

    def forward(self, X):
        return self.inorm(torch.log(X))



m_names = ['Plain',
            'Plain+InCBW0',
            'Plain+InGBW0',
            'RotEq',
            'RotEq+InCBW0',
            'RotEq+InGBW0',
            'ScaleEq',
            'ScaleEq+InCBW0',
            'ScaleEq+InGBW0',
            'SchearEq',
            'SchearEq+InCBW0',
            'SchearEq+InGBW0',
            'GammaEq',
            'BrightnessEq',
            'InB1',
            'InB2',
            'InB3']

def str2arr(s, d):
    arr = s[1:-2].split(',')
    arr = [float(x) for x in arr][:150]
    if d == 'cifar100':
        arr = arr[:100]
    return arr


def plot_temp():
    datasets = ['stl10', 'cifar100', 'cifar10']
    for d in datasets:
        models = os.listdir(f'runs/{d}')
        for m in models:
            with open(f'runs/{d}/{m}', 'r') as f:
                lines = f.readlines()
                tr = lines[1]
                tr = str2arr(tr, d)
                plt.plot(np.arange(len(tr)), tr, label=m)
            plt.legend()
            plt.title(d)
            plt.show()


def plot1():
    model_names = ['Plain', 'RotEq', 'ScaleEq', 'SchearEq', 'GammaEq', 'BrightnessEq']
    datasets = ['stl10', 'cifar100', 'cifar10']
    plt.figure(figsize=(10, 10), dpi=200)
    for i, d in enumerate(datasets):
        plt.subplot(3,2,2*i+1)
        plt.title(d.upper() + ' train')
        for m in model_names:
            with open(f'runs/{d}/{m}_False.txt', 'r') as f:
                lines = f.readlines()
                tr = lines[1]
                tr = str2arr(tr, d)
                plt.plot(np.arange(len(tr))+1, tr, label=m)
        plt.grid()
        plt.legend()
        plt.subplot(3,2,2*i+2)
        plt.title(d.upper() + ' test')
        for m in model_names:
            with open(f'runs/{d}/{m}_False.txt', 'r') as f:
                lines = f.readlines()
                te = lines[3]
                te = str2arr(te, d)
                plt.plot(np.arange(len(te))+1, te, label=m)
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig('plots/plot1/plot.pdf')

def plot2():
    model_names = ['Plain', 'RotEq', 'ScaleEq', 'SchearEq']
    postfix = ['', '+InCBW0', '+InGBW0']
    color = ['blue', 'orange', 'red']
    datasets = ['stl10', 'cifar100', 'cifar10']
    for d in datasets:
        plt.figure(figsize=(10, 7), dpi=200)
        for i, m in enumerate(model_names):
            plt.subplot(2,2,i+1)
            plt.title(m)
            for p, c in zip(postfix, color):
                with open(f'runs/{d}/{m}{p}_False.txt', 'r') as f:
                    lines = f.readlines()
                    tr = lines[1]
                    te = lines[3]
                    tr = str2arr(tr, d)
                    te = str2arr(te, d)
                    plt.plot(np.arange(len(tr))+1, tr, '--', color=c,
                    label=f'{m}{p} train')
                    plt.plot(np.arange(len(te))+1, te, '-', color=c,
                    label=f'{m}{p} test')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        plt.savefig(f'plots/plot2/{d}.pdf')
        plt.clf()
        print(f'{d}.pdf saved')

def plot3():
    model_names = ['BrightnessEq', 'InB1', 'InB2', 'InB3']
    datasets = ['stl10', 'cifar100', 'cifar10']
    plt.figure(figsize=(10, 10), dpi=200)
    for i, d in enumerate(datasets):
        plt.subplot(3,2,2*i+1)
        plt.title(d.upper() + ' train')
        for m in model_names:
            with open(f'runs/{d}/{m}_False.txt', 'r') as f:
                lines = f.readlines()
                tr = lines[1]
                tr = str2arr(tr, d)
                plt.plot(np.arange(len(tr))+1, tr, label=m)
        plt.grid()
        plt.legend()
        plt.subplot(3,2,2*i+2)
        plt.title(d.upper() + ' test')
        for m in model_names:
            with open(f'runs/{d}/{m}_False.txt', 'r') as f:
                lines = f.readlines()
                te = lines[3]
                te = str2arr(te, d)
                plt.plot(np.arange(len(te))+1, te, label=m)
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/plot3/plot.pdf')


def plot_matrix(m, xlabels, ylabels, vmin=None, vmax=None):
    xlabels = [f'{x:.03}' if type(x)==np.float64 else x for x in xlabels]
    plt.imshow(m, vmin=vmin, vmax=vmax)
    plt.xticks(np.arange(len(xlabels)), xlabels)
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.setp(plt.xticks()[1], rotation=45, ha="right",
             rotation_mode="anchor")
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            plt.text(j, i, int(m[i, j]), ha="center", va="center", color="w")

def plot4():
    datasets = ['cifar10', 'cifar100', 'stl10']
    factors = {
            'contrast': 2**np.linspace(np.log2(0.2), np.log2(5), 11),
            'brightness': 2**np.linspace(np.log2(0.2), np.log2(5), 11),
            'color': np.arange(1000, 11001, 1000),
            'gamma': 2**np.linspace(-np.log2(3), np.log2(3), 11)}
    D_NUMS = {d: None for d in datasets}

    for d in D_NUMS:
        f_names = os.listdir(f'results/{d}')
        NUMS = {'contrast': [None for _ in range(17)],
                'brightness': [None for _ in range(17)],
                'color': [None for _ in range(17)],
                'gamma': [None for _ in range(17)]}
        for name in f_names:
            s = name.split('_')
            model_name = s[0]
            transform = s[2].split('.')[0]
            with open(f'results/{d}/{name}', 'r') as f:
                nums = str2arr(f.readlines()[1], d)
                NUMS[transform][m_names.index(model_name)] = nums
        D_NUMS[d] = NUMS

    labels = m_names
    for key in NUMS:
        for i, d in enumerate(D_NUMS):
            plt.figure(figsize=(6.5, 8), dpi=200)
            plot_matrix(np.array(D_NUMS[d][key]), factors[key], labels, vmin=0,
                    vmax=100)
            plt.title(d.upper())
            plt.tight_layout()
            plt.savefig(f'plots/plot4/{key}_{d}.pdf')
            plt.clf()
            print(f'{key}_{d}.pdf saved')

def get_best(dataset):
    dir = f'runs/{dataset}'
    f_names = os.listdir(dir)
    results = {f_name: 0 for f_name in f_names}

    for f_name in f_names:
        with open(os.path.join(dir, f_name), 'r') as f:
            results[f_name] = max(str2arr(f.readlines()[3], dataset))
    best = sorted(list(results.items()), key = lambda nv: nv[1])[-5:]
    return [x[0] for x in best]

def plot5():
    datasets = ['stl10', 'cifar100', 'cifar10']
    plt.figure(figsize=(10, 10), dpi=200)
    for i, d in enumerate(datasets):
        plt.subplot(3,2,2*i+1)
        plt.title(d.upper() + ' train')
        model_names = get_best(d)
        for m in model_names:
            with open(f'runs/{d}/{m}', 'r') as f:
                lines = f.readlines()
                tr = lines[1]
                tr = str2arr(tr, d)
                plt.plot(np.arange(len(tr))+1, tr, label=m)
        plt.grid()
        plt.legend()
        plt.subplot(3,2,2*i+2)
        plt.title(d.upper() + ' test')
        for m in model_names:
            with open(f'runs/{d}/{m}', 'r') as f:
                lines = f.readlines()
                te = lines[3]
                te = str2arr(te, d)
                plt.plot(np.arange(len(te))+1, te, label=m)
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/plot5/plot.pdf')

def plot7():
    plt.rc('font', size=15) #controls default text size
    models = ['BrightnessEq', 'GammaEq', 'RotEq', 'ScaleEq', 'SchearEq']
    transforms = ['brightness', 'gamma', 'rotate', 'scale', 'shear']
    layers = ['conv1', 'bn1', 'ac1', 'maxpool1', 'layer1', 'layer2']
    XLABELS = ['conv1', 'norm1', 'ac1', 'maxpool1', 'bottleneck1', 'bottleneck2']
    file_names = os.listdir('equivariance')
    for t, m in zip(transforms, models):
        plain_dict, eq_dict = {}, {}
        for layer in layers:
            with open(f'equivariance/Plain_{t}_False_{layer}.json', 'r') as f:
                plain_dict[layer] = json.load(f)['err_abs']
            with open(f'equivariance/{m}_{t}_False_{layer}.json', 'r') as f:
                eq_dict[layer] = json.load(f)['err_abs']

        ylabels = list(plain_dict['bn1'].keys())
        m_plain = [[100*plain_dict[layer][v] for v in ylabels] for layer in layers]
        eq_plain = [[100*eq_dict[layer][v] for v in ylabels] for layer in layers]
        ylabels = [f'{float(x):.2f}' for x in plain_dict['bn1'].keys()]

        xlabels = XLABELS[:]
        plt.figure(figsize=(5, 8), dpi=200)
        plot_matrix(np.array(m_plain).T, xlabels, ylabels, 0, 150) 
        plt.title(f'Plain model, {t} equivariance')
        plt.tight_layout()
        plt.savefig(f'plots/plot7/plain_{t}.pdf')
        plt.clf()

        if m != 'BrightnessEq': xlabels[0] = 'lift'
        plt.figure(figsize=(5, 8), dpi=200)
        plot_matrix(np.array(eq_plain).T, xlabels, ylabels, 0, 150) 
        plt.title(f'{m} model, {t} equivariance')
        plt.tight_layout()
        plt.savefig(f'plots/plot7/{m}_{t}.pdf')
        plt.clf()

def plot8():
    plt.rc('font', size=15) #controls default text size
    plt.figure(figsize=(15, 8), dpi=200)
    with open(f'equivariance/bresnet_norms.json', 'r') as f:
        plain_dict = json.load(f)['Plain']
    with open(f'equivariance/bresnet_norms.json', 'r') as f:
        b_dict = json.load(f)['Bresnet']
    labels = list(plain_dict.keys())
    labels = ['conv1']
    for i in range(1,7):
        labels.append(f'bottleneck{i}.conv1')
        labels.append(f'bottleneck{i}.conv2')
    plain_vals = list(plain_dict.values())
    b_vals = list(b_dict.values())
    plt.bar(np.arange(len(plain_vals))+0.125, plain_vals,
            log=True, width=0.25, label='PlainResNet')
    plt.bar(np.arange(len(plain_vals))-0.125, b_vals,
            log=True,width=0.25, label='BResNet')
    plt.grid()
    plt.legend()
    plt.xticks(np.arange(len(plain_vals)), labels)
    plt.setp(plt.xticks()[1], rotation=30, ha="right",
             rotation_mode="anchor")
    plt.tight_layout()
    plt.savefig(f'plots/plot8/norms.pdf')


plot8()

