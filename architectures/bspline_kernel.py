import torch
from torchvision.transforms.functional import center_crop as torch_center_crop
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torchvision

from groups import SO2, Rplus, RplusContrast, Rshear

def B2(x):
    # Approximation of original B2-spline.
    # Found with scipy.curve_fit
    return 2**(-x*x*2.6) / 1.3

    # original B2-spline:
    #return (-3*(-1/2 + x)**2 * torch.sign(1/2 - x) +
    #       (-3/2 + x)**2 * torch.sign(3/2 - x) -
    #       (3*(1 + 2*x)**2 * torch.sign(1/2 + x))/4 +
    #       ((3 + 2*x)**2 * torch.sign(3/2 + x))/4)/4



class Lift(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, N_h, group, stride=1,
            padding=0, mask=False):
        super(Lift, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.C_in = C_in
        self.C_out = C_out

        self.group = group
        self.mask = mask
        self.h_sample_points = self.group.grid(N_h)
        self.raw_spline_vals = self.sample_splines(kernel_size)

        self.weights = nn.Parameter(torch.randn(C_in, kernel_size**2, C_out),
                requires_grad=True)
        #with open('w1.npy', 'rb') as f:
        #    self.weights.data = torch.from_numpy(np.load(f))

    def sample_splines(self, kernel_size):
        low, high = -(kernel_size//2), kernel_size//2
        sample_grid = np.array([[[i,j] for j in range(low,high+1)] for i in
            range(low,high+1)])
        centers = sample_grid.copy().reshape(kernel_size**2,2)
        centers = np.stack([centers]*self.C_in)
        centers = centers.transpose(2,0,1)
        centers = torch.from_numpy(centers)

        raw_spline_vals = {}
        for h in self.h_sample_points:
            xxs = self.group.transform_xx_coordinates(sample_grid, self.group.inv(h))
            xxs = xxs.reshape(kernel_size, kernel_size, 2, 1, 1)
            xxs = torch.from_numpy(xxs)
            diffs = centers - xxs
            vals = torch.prod(B2(diffs), dim=-3).float()
            raw_spline_vals[h] = vals.unsqueeze(-1).cuda()
        return raw_spline_vals

    def kernel(self, h):
        '''
        Return (C_out, C_in, kernel_size, kernel_size)-shaped tensor.
        It's values are sampled at positions corresponding to sample_grid
        rotated by angle theta.
        '''
        vals = self.raw_spline_vals[h]
        _weights = self.weights.unsqueeze(0).unsqueeze(0)
        kernel = (1/self.group.det(h)) * torch.sum(vals * _weights, axis=-2).permute(3,2,0,1)
        return self.group.transform_kernel(kernel, h)

    def forward(self, X):
        kernel_stack = torch.cat([self.kernel(h) for h in self.h_sample_points],
                dim=0)
        #for i in range(kernel_stack.shape[0]):
        #    plt.subplot(4,2,2*i+1)
        #    plt.imshow(torch.abs(kernel_stack[i,0,:,:]).cpu().numpy())
        #    plt.subplot(4,2,2*i+2)
        #    plt.imshow(torch_center_crop(X[0,0], [11,11]).cpu().numpy())
        #plt.show()
        if self.group == SO2 and self.mask:
            kernel_stack[:,:, 0, 0] = 0
            kernel_stack[:,:, 0,-1] = 0
            kernel_stack[:,:,-1, 0] = 0
            kernel_stack[:,:,-1,-1] = 0
        out = F.conv2d(X, kernel_stack, stride=self.stride, padding=self.padding, bias=None)
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
        self.h_sample_grid = self.group.grid(N_h)
        self.raw_spline_vals = self.sample_splines(kernel_size, h_basis_size, C_in)

        self.weights = nn.Parameter(torch.randn(C_in, h_basis_size *
            kernel_size**2, C_out), requires_grad=True)
        #with open('w2.npy', 'rb') as f:
        #    self.weights.data = torch.from_numpy(np.load(f))

    def sample_splines(self, kernel_size, h_basis_size, C_in):
        # construct centers of splines
        low, high = -(kernel_size//2), kernel_size//2
        xx_sample_grid = np.array([[[i,j] for j in range(low,high+1)] for i in
            range(low,high+1)], dtype=np.float32)
        xx_centers = xx_sample_grid.copy().reshape(kernel_size**2,2)
        xx_centers = np.stack([xx_centers]*C_in)
        xx_centers = np.repeat(xx_centers, h_basis_size, axis=1)
        xx_centers = xx_centers.transpose(2,0,1)
        xx_centers = torch.from_numpy(xx_centers)

        h_centers = self.group.grid(h_basis_size)
        h_centers = np.concatenate([h_centers] * kernel_size**2)
        h_centers = np.stack([h_centers]*C_in)
        h_centers = np.expand_dims(h_centers, axis=2)
        h_centers = torch.from_numpy(h_centers)

        # compute raw spline values for all h in h_sample_grid
        raw_spline_vals = {}
        for h in self.h_sample_grid:
            # sample kernel values on spatial dimensions
            xxs = self.group.transform_xx_coordinates(xx_sample_grid,
                    self.group.inv(h))
            xxs = torch.from_numpy(xxs.reshape(self.kernel_size, self.kernel_size,
                2, 1, 1))
            xx_diffs = xx_centers - xxs
            xx_vals = torch.prod(B2(xx_diffs), dim = -3)

            # sample kernel values on rotational dimension
            hs = self.group.prod(self.h_sample_grid, self.group.inv(h))
            hs = torch.from_numpy(hs)
            h_diffs = torch.stack([self.group.dist(hh, h_centers)
                        for hh in hs], dim=0)[...,0]
            h_vals = B2(h_diffs / self.h_scale)

            # multiply both
            xx_vals = xx_vals.unsqueeze(2)
            h_vals = h_vals.unsqueeze(0).unsqueeze(0)
            vals = xx_vals * h_vals
            vals = vals.unsqueeze(-1)
            raw_spline_vals[h] = vals.cuda()
        return raw_spline_vals


    def kernel(self, h):
        '''
        Return (C_out, C_in, N_rot, kernel_size, kernel_size)-shaped tensor.
        It's values are sampled at positions corresponding to xx_sample_grid
        and rot_sample_grid rotated by angle theta.
        '''
        vals = self.raw_spline_vals[h]
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

class Projection(nn.Module):
    def __init__(self):
        super(Projection, self).__init__()

    def forward(self, X):
        return X.sum(dim=2)


def value_test():
    l1 = Lift(C_out=3, kernel_size=5, N_h=7, group=SO2).cuda()
    o1 = l1(torch.ones(1,3,14,14).cuda())
    print('o1:')
    print(o1.sum(dim=(0,2,3,4)))
    print(o1.sum(dim=(0,1,3,4)))

    l2 = LiftedConv(C_in = 3, C_out=4, kernel_size=5, N_h=7,
            h_basis_size=13, group=SO2).cuda()
    o2 = l2(torch.ones(1,3,7,10,10).cuda())
    print('\no2:')
    print(o2.sum(dim=(0,2,3,4)))
    print(o2.sum(dim=(0,1,3,4)))

def equivariance_test():
    with torch.no_grad():
        X = torch.from_numpy(plt.imread('pic.jpeg')).cuda().unsqueeze(0)
        X = X.permute([0,3,1,2])#[:,:3,:,:]
        X = X.float() / 255.0
        #X += torch.randn(X.shape).cuda()*0.1
        #X = torch.rand(1,3,105,105).cuda()
        #aa = np.sin(np.arange(100))*2-1
        #aa = torch.from_numpy(np.outer(aa,aa)).cuda().float()
        #X = torch.stack([aa]*3).unsqueeze(0)
        #X = torch.rand(1,3,100,100).cuda()

        h = 1
        ROLL = 0
        N_h = 1
        G = Rshear
        SHOW = True
        l1 = Lift(C_in=3, C_out=3, kernel_size=5, N_h=N_h, group=G).cuda()
        l2 = LiftedConv(C_in=3, C_out=3, kernel_size=5, N_h=N_h, h_basis_size=N_h, group=G).cuda()
        l1 = nn.Conv2d(3, 3, 5).cuda()
        l2 = nn.Conv2d(3, 3, 5).cuda()
        #l2 = nn.Identity()
        o1 = G.transform_tensor(l2(l1(X)), h).cpu()
        o2 = l2(l1(G.transform_tensor(X, h))).cpu()
        o2 = torchvision.transforms.functional.center_crop(o2, o1.shape[-2:])
        o1 = o1.unsqueeze(2)
        o2 = o2.unsqueeze(2)
        o1 = o1.numpy()
        o2 = o2.numpy()

        # value at particular position before rolling in depth dimension:
        x = np.arange(N_h)
        pos = X.shape[-1]//2
        y1 = o1[0,0,x,pos,pos]
        y2 = o2[0,0,x,pos,pos]
        plt.plot(x,y1,label='o1')
        plt.plot(x,y2,label='o2')
        plt.legend()
        plt.show()

        # 0th channel maps before rolling in depth dimension
        if SHOW:
            for i in x:
                plt.subplot(2,len(x),i+1)
                plt.imshow((np.tanh(o1[0,:,i,:,:]/10)/2+1/2).transpose(1,2,0).clip(0,1))
                plt.subplot(2,len(x),i+1+len(x))
                plt.imshow((np.tanh(o2[0,:,i,:,:]/10)/2+1/2).transpose(1,2,0).clip(0,1))
            plt.show()

        #roll:
        o1 = np.roll(o1, ROLL, axis=2)
        print('Absolute diff:', np.abs(o1-o2).mean(axis=(0,1,3,4)))
        print('Relative diff:', (np.abs(o1-o2)/
            (np.abs(o1)+np.abs(o2)/2+1e-6)).mean(axis=(0,1,3,4)))

        if SHOW:
            for i in x:
                plt.subplot(1,len(x),i+1)
                plt.imshow(np.abs(o1[0,0,i,:,:]-o2[0,0,i,:,:]))
            plt.legend()
            plt.show()

    #l2 = LiftedConv(C_in = 3, C_out=4, kernel_size=5, N_h=7,
    #        h_basis_size=13, group=SO2).cuda()

if __name__ == '__main__':
    equivariance_test()
