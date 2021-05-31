import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from groups import SO2, Rplus

def B2(x):
    return (-3*(-1/2 + x)**2 * torch.sign(1/2 - x) +
           (-3/2 + x)**2 * torch.sign(3/2 - x) -
           (3*(1 + 2*x)**2 * torch.sign(1/2 + x))/4 +
           ((3 + 2*x)**2 * torch.sign(3/2 + x))/4)/4


class Lift(nn.Module):
    def __init__(self, C_out, kernel_size, N_h, group, stride=1, mask=False):
        super(Lift, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.C_out = C_out
        self.group = group
        self.mask = mask

        l, h = -(kernel_size//2), kernel_size//2
        self.sample_grid = np.array([[[i,j] for j in range(l,h+1)] for i in
            range(l,h+1)])
        self.centers = self.sample_grid.copy().reshape(25,2)
        self.centers = np.stack([self.centers]*3)
        self.centers = self.centers.transpose(2,0,1)
        self.centers = torch.from_numpy(self.centers).cuda()
        self.h_sample_points = self.group.grid(N_h)

        self.weights = nn.Parameter(torch.randn(3, kernel_size**2, C_out),
                requires_grad=True)

    def kernel(self, h):
        '''
        Return (C_out, C_in, kernel_size, kernel_size)-shaped tensor.
        It's values are sampled at positions corresponding to sample_grid
        rotated by angle theta.
        '''
        xxs = self.group.transform_xx_coordinates(self.sample_grid,
                self.group.inv(h))
        xxs = xxs.reshape(self.kernel_size, self.kernel_size, 2, 1, 1)
        xxs = torch.from_numpy(xxs).cuda()
        diffs = self.centers - xxs
        vals = torch.prod(B2(diffs), dim=-3)
        vals = vals.unsqueeze(-1)
        _weights = self.weights.unsqueeze(0).unsqueeze(0)
        return torch.sum(vals * _weights, axis=-2).permute(3,2,0,1).float()

    def forward(self, X):
        kernel_stack = torch.cat([self.kernel(h) for h in self.h_sample_points],
                dim=0)
        if self.group == SO2 and self.mask:
            kernel_stack[:,:, 0, 0] = 0
            kernel_stack[:,:, 0,-1] = 0
            kernel_stack[:,:,-1, 0] = 0
            kernel_stack[:,:,-1,-1] = 0
        out = F.conv2d(X, kernel_stack, stride=self.stride, padding=0, bias=None)
        b, cd, h, w = out.shape
        out = out.reshape(b, len(self.h_sample_points), self.C_out, h, w)
        out = out.transpose(1,2)
        return out


class LiftedConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, N_h, h_basis_size, group,
            stride=1, padding=0, mask=False):
        super(LiftedConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.C_out = C_out
        self.group = group
        self.mask = mask

        self.h_scale = self.group.scale(h_basis_size)

        l, h = -(kernel_size//2), kernel_size//2
        self.xx_sample_grid = np.array([[[i,j] for j in range(l,h+1)] for i in
            range(l,h+1)], dtype=np.float32)
        self.xx_centers = self.xx_sample_grid.copy().reshape(25,2)
        self.xx_centers = np.stack([self.xx_centers]*C_in)
        self.xx_centers = np.repeat(self.xx_centers, h_basis_size, axis=1)
        self.xx_centers = self.xx_centers.transpose(2,0,1)
        self.xx_centers = torch.from_numpy(self.xx_centers).cuda()

        self.h_sample_grid = self.group.grid(N_h)
        self.h_centers = self.group.grid(h_basis_size)
        self.h_centers = np.concatenate([self.h_centers] * kernel_size**2)
        self.h_centers = np.stack([self.h_centers]*C_in)
        self.h_centers = np.expand_dims(self.h_centers, axis=2)
        self.h_centers = torch.from_numpy(self.h_centers).cuda()

        self.weights = nn.Parameter(torch.randn(C_in, h_basis_size *
            kernel_size**2, C_out), requires_grad=True)
        #with open('w.npy', 'rb') as f:
        #    self.weights.data = torch.from_numpy(np.load(f))

    def kernel(self, h):
        '''
        Return (C_out, C_in, N_rot, kernel_size, kernel_size)-shaped tensor.
        It's values are sampled at positions corresponding to xx_sample_grid
        and rot_sample_grid rotated by angle theta.
        '''
        # sample kernel values on spatial dimensions
        xxs = self.group.transform_xx_coordinates(self.xx_sample_grid,
                self.group.inv(h))
        xxs = torch.from_numpy(xxs.reshape(self.kernel_size, self.kernel_size,
            2, 1, 1)).cuda()
        xx_diffs = self.xx_centers - xxs
        xx_vals = torch.prod(B2(xx_diffs), dim = -3)
        print(xx_vals.shape)

        # sample kernel values on rotational dimension
        hs = self.group.prod(self.h_sample_grid, self.group.inv(h))
        hs = torch.from_numpy(hs).cuda()
        h_diffs = torch.stack([self.group.dist(hh, self.h_centers)
                    for hh in hs], dim=0)[...,0]
        h_vals = B2(h_diffs / self.h_scale)

        xx_vals = xx_vals.unsqueeze(2)
        h_vals = h_vals.unsqueeze(0).unsqueeze(0)
        vals = xx_vals * h_vals
        vals = vals.unsqueeze(-1)
        _weights = self.weights.reshape([1,1,1]+[*self.weights.shape])
        return (1/self.group.det(h)) * torch.sum(vals*_weights, axis=-2).permute(4,3,2,0,1)

    def forward(self, X):
        kernel_stack = torch.cat([self.kernel(h)
            for h in self.h_sample_grid], dim=0)
        if self.group == SO2 and self.mask:
            kernel_stack[..., 0, 0] = 0
            kernel_stack[..., 0,-1] = 0
            kernel_stack[...,-1, 0] = 0
            kernel_stack[...,-1,-1] = 0
        n, c, d, h, w = kernel_stack.shape
        kernel_stack = kernel_stack.reshape(n, c*d, h, w)
        n, c, d, h, w = X.shape
        X = X.reshape(n, c*d, h, w)
        out = F.conv2d(X, kernel_stack, stride=self.stride, padding=self.padding, bias=None)
        n, _, h, w = out.shape
        out = out.reshape(n, len(self.h_sample_grid), self.C_out, h, w)
        out = out.transpose(1,2)
        return out

import time
l1 = Lift(C_out=3, kernel_size=5, N_h=7, group=SO2).cuda()
a  = time.time()
o1 = l1(torch.ones(1,3,14,14).cuda())
print(time.time()-a)
l2 = LiftedConv(C_in = 16, C_out=32, kernel_size=5, N_h=4,
        h_basis_size=4, group=Rplus).cuda()
a  =time.time()
o2 = l2(torch.ones(16,16,4,100,100).cuda())
print(time.time()-a)
