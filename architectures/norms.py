import torch
from torch import nn

class LogNorm(nn.Module):
    def __init__(self):
        super(LogNorm, self).__init__()
        self.inorm = nn.InstanceNorm2d(3, affine=False, track_running_stats=False, eps=0)

    def forward(self, X):
        return self.inorm(torch.log(X))

class MeanNorm2d(nn.Module):
    def __init__(self, cin):
        super(MeanNorm2d, self).__init__()

    def forward(self, X):   # X is [B,C,D,H,W]
        return X - torch.mean(X, dim=(-3,-2,-1), keepdim=True)

def IN2d(cin):
    return nn.InstanceNorm2d(cin, affine=False, track_running_stats=False,
            eps=1e-6)

