import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def B2(x):
    return (-3*(-1/2 + x)**2 * torch.sign(1/2 - x) +
           (-3/2 + x)**2 * torch.sign(3/2 - x) -
           (3*(1 + 2*x)**2 * torch.sign(1/2 + x))/4 +
           ((3 + 2*x)**2 * torch.sign(3/2 + x))/4)/4

def rotate_xx_coordinates(grid, theta):
    x = grid[...,0]
    y = grid[...,1]
    x_new = x*np.cos(theta) - y*np.sin(theta)
    y_new = x*np.sin(theta) + y*np.cos(theta)
    return np.stack([x_new,y_new], axis=-1)

class Lift(nn.Module):
    def __init__(self, C_out, kernel_size, N_rot, stride=1, mask=False):
        super(Lift, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.C_out = C_out

        l, h = -(kernel_size//2), kernel_size//2
        self.sample_grid = np.array([[[i,j] for j in range(l,h+1)] for i in
            range(l,h+1)])
        self.centers = self.sample_grid.copy().reshape(25,2)
        self.centers = np.stack([self.centers]*3)
        self.centers = self.centers.transpose(2,0,1)
        self.centers = torch.from_numpy(self.centers).cuda()
        self.rotations = np.linspace(0, 2*np.pi, N_rot, endpoint=False)

        self.weights = nn.Parameter(torch.randn(3, kernel_size**2, C_out),
                requires_grad=True)

    def kernel(self, theta):
        '''
        Return (C_out, C_in, kernel_size, kernel_size)-shaped tensor.
        It's values are sampled at positions corresponding to sample_grid
        rotated by angle theta.
        '''
        xxs = rotate_xx_coordinates(self.sample_grid, theta)
        xxs = xxs.reshape(self.kernel_size, self.kernel_size, 2, 1, 1)
        xxs = torch.from_numpy(xxs).cuda()
        diffs = self.centers - xxs
        vals = torch.prod(B2(diffs), dim=-3)
        vals = vals.unsqueeze(-1)
        _weights = self.weights.unsqueeze(0).unsqueeze(0)
        return torch.sum(vals * _weights, axis=-2).permute(3,2,0,1).float()

    def forward(self, X):
        kernel_stack = torch.cat([self.kernel(-theta) for theta in self.rotations],
                axis=0)
        out = F.conv2d(X, kernel_stack, stride=self.stride, padding=0, bias=None)
        b, cd, h, w = out.shape
        out = out.reshape(b, len(self.rotations), self.C_out, h, w)
        out = out.transpose(1,2)
        return out


class LiftedConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, N_rot, rot_basis_size,
            stride=1, padding=0, mask=False):
        super(LiftedConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.C_out = C_out

        self.rot_scale = 2*np.pi/rot_basis_size

        l, h = -(kernel_size//2), kernel_size//2
        self.xx_sample_grid = np.array([[[i,j] for j in range(l,h+1)] for i in
            range(l,h+1)], dtype=np.float32)
        self.xx_centers = self.xx_sample_grid.copy().reshape(25,2)
        self.xx_centers = np.stack([self.xx_centers]*C_in)
        self.xx_centers = np.repeat(self.xx_centers, rot_basis_size, axis=1)
        self.xx_centers = self.xx_centers.transpose(2,0,1)
        self.xx_centers = torch.from_numpy(self.xx_centers).cuda()

        self.rot_sample_grid = np.linspace(0, 2*np.pi, N_rot, endpoint=False,
                dtype=np.float32)
        self.rot_centers = np.linspace(0, 2*np.pi, rot_basis_size,
                endpoint=False, dtype=np.float32)
        self.rot_centers = np.concatenate([self.rot_centers] * kernel_size**2)
        self.rot_centers = np.stack([self.rot_centers]*C_in)
        self.rot_centers = np.expand_dims(self.rot_centers, axis=2)
        self.rot_centers = torch.from_numpy(self.rot_centers).cuda()

        self.weights = nn.Parameter(torch.randn(C_in, rot_basis_size *
            kernel_size**2, C_out), requires_grad=True)
        #with open('w.npy', 'rb') as f:
        #    self.weights.data = torch.from_numpy(np.load(f))

    def angle_dist(self, th1, th2):
        return torch.remainder(th2-th1+np.pi, 2*np.pi) - np.pi

    def kernel(self, theta):
        '''
        Return (C_out, C_in, N_rot, kernel_size, kernel_size)-shaped tensor.
        It's values are sampled at positions corresponding to xx_sample_grid
        and rot_sample_grid rotated by angle theta.
        '''
        # sample kernel values on spatial dimensions
        xxs = rotate_xx_coordinates(self.xx_sample_grid, theta)
        xxs = torch.from_numpy(xxs.reshape(self.kernel_size, self.kernel_size,
            2, 1, 1)).cuda()
        xx_diffs = self.xx_centers - xxs
        xx_vals = torch.prod(B2(xx_diffs), dim = -3)

        # sample kernel values on rotational dimension
        rots = self.rot_sample_grid + theta
        rots = torch.from_numpy(rots).cuda()
        rot_diffs = torch.stack([self.angle_dist(r, self.rot_centers)
                    for r in rots], dim=0)[...,0]
        rot_vals = B2(rot_diffs / self.rot_scale)

        xx_vals = xx_vals.unsqueeze(2)
        rot_vals = rot_vals.unsqueeze(0).unsqueeze(0)
        vals = xx_vals * rot_vals
        vals = vals.unsqueeze(-1)
        _weights = self.weights.reshape([1,1,1]+[*self.weights.shape])
        return torch.sum(vals*_weights, axis=-2).permute(4,3,2,0,1)

    def forward(self, X):
        kernel_stack = torch.cat([self.kernel(-theta)
            for theta in self.rot_sample_grid], dim=0)
        n, c, d, h, w = kernel_stack.shape
        kernel_stack = kernel_stack.reshape(n, c*d, h, w)
        n, c, d, h, w = X.shape
        X = X.reshape(n, c*d, h, w)
        out = F.conv2d(X, kernel_stack, stride=self.stride, padding=self.padding, bias=None)
        n, _, h, w = out.shape
        out = out.reshape(n, len(self.rot_sample_grid), self.C_out, h, w)
        out = out.transpose(1,2)
        out = (2*np.pi / len(self.rot_sample_grid)) * out
        return out

l1 = Lift(C_out=3, kernel_size=5, N_rot=7).cuda()
o1 = l1(torch.ones(1,3,14,14).cuda())
l2 = LiftedConv(C_in = 5, C_out=9, kernel_size=5, N_rot=7,
        rot_basis_size=13).cuda()
o2 = l2(torch.ones(1,5,7,10,10).cuda())
print(o2.sum(dim=(0,2,3,4)))
print(o2.sum(dim=(0,1,3,4)))
