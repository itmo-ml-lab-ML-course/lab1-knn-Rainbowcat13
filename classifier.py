from kernels import Uniform
from metrics import Euclid

INF = 100


class KNNClassifier:
    def __init__(self, n=5, kernel=Uniform(), metric=Euclid(), fixed_window_size=None):
        self.n = n
        self.kernel = kernel.calc_k
        self.metric = metric.calculate_metric
        self.h = fixed_window_size
        self.X = None
        self.y = None
        self.classes = None

    def fit(self, X, y):
        self.X = X.tolist()
        self.y = y.tolist()
        self.classes = {i: cy for i, (_, cy) in enumerate(zip(self.X, self.y))}

    def predict(self, X):
        result = []

        for idx, x in enumerate(X):
            # print(f'{idx}/{len(X)}')
            distances = []
            for i, tx in enumerate(self.X):
                distances.append((self.metric(x, tx), i))

            distances.sort(key=lambda t: t[0])

            for i, d in enumerate(distances):
                if self.h is not None:
                    distances[i] = (d[0] / self.h, d[1])
                else:
                    distances[i] = (d[0] / distances[self.n][0], d[1])

            argmax = 0
            cls = None
            mapcls = dict()
            for d in distances:
                if mapcls.get(d[1]) is None:
                    mapcls[d[1]] = self.kernel(d[0])
                else:
                    mapcls[d[1]] += self.kernel(d[0])

            for xx, metric in mapcls.items():
                if metric > argmax:
                    argmax = metric
                    cls = self.classes[xx]

            result.append(cls or self.y[0])

        return result
