from abc import ABC, abstractmethod
import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import rotate as torch_rotate
from torchvision.transforms.functional import resize as torch_resize
from torchvision.transforms.functional import center_crop as torch_center_crop
from torchvision.transforms.functional import adjust_contrast as torch_adjust_contrast
from torchvision.transforms.functional import affine as torch_affine
import PIL


class Group(ABC):
    @abstractmethod
    def inv(h):
        pass

    @abstractmethod
    def transform_xx_coordinates(grid, h):
        pass

    @abstractmethod
    def grid(N_h):
        pass

    @abstractmethod
    def scale(h_basis_size):
        pass

    @abstractmethod
    def prod(h1, h2):
        pass

    @abstractmethod
    def dist(th1, th2):
        pass

    @abstractmethod
    def det(h):
        pass

    @abstractmethod
    def transform_tensor(X, h):
        pass

    @abstractmethod
    def transform_kernel(kernel, h):
        pass

    @abstractmethod
    def activation():
        pass


class SO2(Group):
    def inv(theta):
        return -theta

    def transform_xx_coordinates(grid, theta):
        x = grid[...,0]
        y = grid[...,1]
        x_new = x*np.cos(theta) - y*np.sin(theta)
        y_new = x*np.sin(theta) + y*np.cos(theta)
        return np.stack([x_new,y_new], axis=-1)

    def grid(N_h):
        return np.linspace(0, 2*np.pi, N_h, endpoint=False, dtype=np.float32)

    def scale(h_basis_size):
        return 2*np.pi/h_basis_size

    def prod(th1, th2):
        return th1 + th2

    def dist(th1, th2):
        return torch.remainder(th2-th1+np.pi, 2*np.pi) - np.pi

    def det(theta):
        return 1

    def transform_tensor(X, rad_angle):
        '''
        rad_angle is angle in radians.
        '''
        angle = int(rad_angle * 180/np.pi)
        if len(X.shape) == 4:
            return torch_rotate(X, angle, resample=PIL.Image.BILINEAR,
                    expand=True)
        elif len(X.shape) == 5:
            I = np.arange(X.shape[2])
            out = torch.stack([torch_rotate(X[:,:,i,:,:],
                        angle, resample=PIL.Image.BILINEAR, expand=True)
                        for i in I], dim=2)
            return out
        else:
            raise Exception("Wrong X shape")

    def transform_kernel(kernel, h):
        return kernel

    def activation():
        return nn.ReLU()

class Rplus(Group):
    def inv(s):
        return 1/s

    def transform_xx_coordinates(grid, s):
        return grid*s

    def grid(N):
        return np.array([np.sqrt(2)**i for i in range(N)], dtype=np.float32)

    def scale(h_basis_size):
        return np.log(2)/2

    def prod(s1, s2):
        return s1 * s2

    def dist(s1, s2):
        return torch.log(s2/s1)

    def det(s):
        return s**2

    def transform_tensor(X, scale):
        h, w = X.shape[-2:]
        if len(X.shape) == 4:
            return torch_affine(x, 0, (0,0), scale, (0,0), resample=PIL.Image.BILINEAR)
        elif len(X.shape) == 5:
            I = np.arange(X.shape[2])
            out = torch.stack([Rplus.transform_tensor(X[:,:,i,:,:], scale)
                        for i in I], dim=2)
            return out
        else:
            raise Exception("Wrong X shape")

    def transform_kernel(kernel, h):
        return kernel

    def activation():
        return nn.ReLU()

class RplusContrast(Group):
    def inv(s):
        return 1/s

    def transform_xx_coordinates(grid, s):
        return grid

    def grid(N):
        return np.array([np.sqrt(2)**i for i in range(-(N//2), N//2+1)], dtype=np.float32)

    def scale(h_basis_size):
        return np.log(2)/2

    def prod(s1, s2):
        return s1 * s2

    def dist(s1, s2):
        return torch.log(s2/s1)

    def det(s):
        return 1

    def transform_tensor(X, factor):    # X is either [B,C,D,H,W] or [B,C,H,W]
        if len(X.shape) == 4:
            dim = (-3,-2,-1)
        elif len(X.shape) == 5:
            dim = (-4,-2,-1)
        else:
            raise Exception('Wrong dimensionality of input tensor')
        mean = torch.mean(X, dim=dim, keepdim=True)
        return X*factor + (1-factor)*mean

    def transform_kernel(kernel, factor): #[N_out, N_in, H, W]
        return RplusContrast.transform_tensor(kernel, factor)

    def activation():
        return nn.Softsign()

class Rshear(Group):
    def inv(x):
        return -x

    def transform_xx_coordinates(grid, a):
        x = grid[...,0]
        y = grid[...,1]
        x_new = x + a*y
        y_new = y
        return np.stack([x_new,y_new], axis=-1)

    def grid(N):
        return np.linspace(-1, 1, N, dtype=np.float32)

    def scale(h_basis_size):
        return 1 / (h_basis_size//2)

    def prod(x1, x2):
        return x1 + x2

    def dist(x1, x2):
        return x2 - x1

    def det(x):
        return 1

    def transform_tensor(X, shear):
        if len(X.shape) == 4:
            shear_angle = np.arctan(shear) * 180/np.pi
            return torch_affine(X, angle=0, translate=[0,0], scale=1,
                    shear=shear_angle)
        elif len(X.shape) == 5:
            I = np.arange(X.shape[2])
            out = torch.stack([Rshear.transform_tensor(X[:,:,i,:,:], shear)
                        for i in I], dim=2)
            return out
        else:
            raise Exception("Wrong X shape")

    def transform_kernel(kernel, h):
        return kernel

    def activation():
        return nn.ReLU()

class RplusGamma(Group):
    def inv(s):
        return 1/s

    def transform_xx_coordinates(grid, s):
        return grid

    def grid(N):
        return np.array([np.sqrt(2)**i for i in range(-(N//2), N//2+1)], dtype=np.float32)

    def scale(h_basis_size):
        return np.log(2)/2

    def prod(s1, s2):
        return s1 * s2

    def dist(s1, s2):
        return torch.log(s2/s1)

    def det(s):
        return 1

    def transform_tensor(X, factor):
        return X.sign() * X.abs()**factor

    def transform_kernel(kernel, factor): #[N_out, N_in, H, W]
        return RplusGamma.transform_tensor(kernel, factor)

    def activation():
        return nn.Softsign()


