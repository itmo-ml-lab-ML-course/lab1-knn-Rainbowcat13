import math
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def calculate(self, x, y):
        if not self.check(x, y):
            raise ValueError("Length of parameters differs")
        return self.calculate_metric(x, y)

    @abstractmethod
    def calculate_metric(self, x, y):
        pass

    def check(self, x, y):
        return len(x) == len(y)


class Euclid(BaseMetric):
    def calculate_metric(self, x, y):
        return math.sqrt(sum((x[i] - y[i]) ** 2 for i in range(len(x))))


class Manhattan(BaseMetric):
    def calculate_metric(self, x, y):
        return sum(abs(x[i] - y[i]) for i in range(len(x)))


class Cosine(BaseMetric):
    def calculate_metric(self, x, y):
        return (1 - sum(x[i] * y[i] for i in range(len(x))) /
                (math.sqrt(sum(x[i] ** 2 for i in range(len(x)))) *
                 math.sqrt(sum(y[i] ** 2 for i in range(len(y))))))


def metric(metric_name):
    if metric_name == 'euclid':
        return Euclid()
    elif metric_name == 'manhattan':
        return Manhattan()
    elif metric_name == 'cosine':
        return Cosine
    else:
        raise ValueError('Unknown metric name')