from abc import ABC, abstractmethod
import numpy as np
import torch

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


