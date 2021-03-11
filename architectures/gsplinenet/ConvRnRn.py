import torch
import bsplines
from torch import nn
import torch.nn.functional as F
import numpy as np
import importlib

class ConvRnRnLayer(nn.Module):
    def __init__(
            self, 
            group,
            C_in,
            C_out,
            kernel_size,
            xx_basis_size=None,
            xx_basis_scale=1,
            xx_basis_type='B_2',
            xx_basis_mask=False,
            stride=1,
            padding=0):
        super(ConvRnRnLayer, self).__init__()

        self.kernel_type = 'Rn'
        self.group = group
        self.G = group.G
        self.H = group.H
        self.Rn = group.Rn

        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.xx_basis_size = xx_basis_size if xx_basis_size is not None else kernel_size
        self.xx_basis_scale = xx_basis_scale
        self.xx_basis_type = xx_basis_type
        self.xx_basis_mask = xx_basis_mask
        self.stride = stride
        self.padding = padding

        self.grid = torch.tensor(self._construct_grid(kernel_size)) # sampling grid
        self.splines = bsplines.B(2, scale=xx_basis_scale) # B-spline
        self.spline_centers = self._construct_spline_centers()
        self.N_k = int(self.spline_centers.shape[1])
        self.weights = self._construct_weights()

    def _construct_grid(self, ker_size, flatten=False, scale=1):
        xx_max= (ker_size - 1)/2
        grid = np.moveaxis(np.mgrid[tuple([slice(-xx_max,xx_max+1)]*self.Rn.n)],0,-1).astype(np.float32)
        if flatten:
            grid = np.reshape(grid,[-1,self.Rn.n])
        return scale*grid

    def _construct_spline_centers(self):
        grid = self._construct_grid(self.xx_basis_size, flatten=True)
        init = np.repeat(grid[np.newaxis,...], self.C_in, 0)
        xx_centers = torch.tensor(init)
        return xx_centers

    def _construct_weights(self):
        # For each input channel, for each basis function, for each output channel a weight
        # So this returns a 3D array
        weights = torch.randn([self.C_in, self.N_k, self.C_out], requires_grad=True)
        n_in = self.C_in * self.N_k
        with torch.no_grad():
            weights *= np.sqrt(2.0 / n_in)
        return weights

    def forward(self, X):
        h = self.H.e
        n = self.Rn.n
        vectors_to_centers = self._Rn_vectors_to_centers(self.grid , self.spline_centers) # For each provided coordinate the distances
        xx_B_sampled = torch.prod(self.splines(vectors_to_centers), dim=-3)

        ## Reshape the weights
        weights_shape = list(self.weights.shape)
        _weights = torch.reshape(self.weights, [1]*(n) + weights_shape)
        xx_B_sampled = xx_B_sampled.unsqueeze(-1)
        print(_weights.shape, xx_B_sampled.shape)
        kernel = torch.sum(xx_B_sampled*_weights, dim=-2) # Sum over the splines
        kernel /= self.H.det(h)
        kernel = kernel.permute(3,2,0,1)
        return F.conv2d(X, kernel, stride=self.stride, padding=self.padding)

    def _Rn_vectors_to_centers(self, xx, xx_centers):
        xx_dim = len(xx.shape)
        xx_centers_dim = len(xx_centers.shape)
        _xx_centers = xx_centers.permute(tuple(np.roll(range(xx_centers_dim),1)))
        _xx = torch.reshape(xx, [xx.shape[i] for i in range(xx_dim)] + [1]*(xx_centers_dim-1))
        differences = _xx_centers - _xx
        return differences

if __name__ == '__main__':
    group = importlib.import_module('group.R2R+')
    l = ConvRnRnLayer(group, 8, 16, 3)
    X = torch.randn((1,8,64,64))
    y =  l(X)
    print(y.shape)