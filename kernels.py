import math
from abc import ABC, abstractmethod


class BaseKernel(ABC):
    def calc_k(self, d):
        if d <= -1 or d >= 1:
            return 0
        return self.k(d)

    @abstractmethod
    def k(self, d):
        pass


class Gaussian(BaseKernel):
    def k(self, d):
        return 1 / math.sqrt(2 * math.pi) * math.exp(-d ** 2 / 2)


class Uniform(BaseKernel):
    def k(self, d):
        return 0.5


class Triangular(BaseKernel):
    def k(self, d):
        return 1 - abs(d)


class Tricube(BaseKernel):
    def k(self, d):
        return 70 / 81 * (1 - abs(d) ** 3) ** 3


def kernel(kernel_name):
    if kernel_name == 'gaussian':
        return Gaussian()
    elif kernel_name == 'uniform':
        return Uniform()
    elif kernel_name == 'triangular':
        return Triangular()
    elif kernel_name == 'tricube':
        return Tricube()
    else:
        raise ValueError('Unknown kernel name')
