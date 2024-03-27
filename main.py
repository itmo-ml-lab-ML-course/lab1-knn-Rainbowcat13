import matplotlib
from sklearn.neighbors import KNeighborsClassifier

from classifier import KNNClassifier
from prepare_data import train_test_val_data
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from kernels import Uniform, Gaussian, Triangular, Tricube
from metrics import Euclid, Manhattan, Cosine
import numpy as np


matplotlib.use('TkAgg')

X_train, y_train, X_val, y_val, X_test, y_test = [x[:15] for x in train_test_val_data()]

mx = -1
accuracies_val, accuracies_train, accuracies_test = [], [], []

ns = list(range(1, 9))
kernels_custom = [Gaussian(), Uniform(), Triangular(), Tricube()]
metrics_custom = [Euclid(), Manhattan(), Cosine()]
wss_custom = [None] + [0.1 * x for x in range(1, 9)]

opt_params_custom = None
for n_neighbors in ns:
    for kernel in kernels_custom:
        for metric in metrics_custom:
            for ws in wss_custom:
                print(f'Neighbors: {n_neighbors}, kernel: {kernel}, metric: {metric}, window size: {ws or "None"}')
                model = KNNClassifier(n=n_neighbors, kernel=kernel, metric=metric, fixed_window_size=ws)

                model.fit(X_train, y_train)
                prediction = model.predict(X_val)

                cur = accuracy_score(y_val, prediction)
                if cur > mx:
                    mx = cur
                    opt_params_custom = [n_neighbors, kernel, metric, ws]

print(opt_params_custom)
print(mx)

mx = -1
kernels = ['uniform', 'distance']
metrics = ['minkowski', 'cityblock', 'cosine']

opt_params = None
for n_neighbors in ns:
    for kernel in kernels:
        for metric in metrics:
            print(f'Neighbors: {n_neighbors}, kernel: {kernel}, metric: {metric}')
            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=kernel, metric=metric)

            model.fit(X_train, y_train)
            prediction = model.predict(X_val)

            cur = accuracy_score(y_val, prediction)
            if cur > mx:
                mx = cur
                opt_params = [n_neighbors, kernel, metric]

print(opt_params)
print(mx)

mx, mx_custom = -1, -1
k_opt, k_opt_custom = 0, 0
parameters = []
accuracies_val, accuracies_train, accuracies_test = [], [], []
accuracies_val_custom, accuracies_train_custom, accuracies_test_custom = [], [], []
for n_neighbors in range(1, 10):
    print(f'Neighbors: {n_neighbors}')
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    custom_model = KNNClassifier(n=n_neighbors)

    model.fit(X_train, y_train)
    custom_model.fit(X_train, y_train)

    custom_prediction = custom_model.predict(X_val)
    prediction = model.predict(X_val)

    cur = accuracy_score(y_val, prediction)
    cur_custom = accuracy_score(y_val, custom_prediction)
    if cur_custom > mx_custom:
        mx_custom = cur_custom
        k_opt_custom = n_neighbors
    if cur > mx:
        mx = cur
        k_opt = n_neighbors

    parameters.append(n_neighbors)
    accuracies_val.append(cur)
    accuracies_train.append(accuracy_score(y_train, model.predict(X_train)))
    accuracies_test.append(accuracy_score(y_test, model.predict(X_test)))

    accuracies_val_custom.append(cur_custom)
    accuracies_train_custom.append(accuracy_score(y_train, custom_model.predict(X_train)))
    accuracies_test_custom.append(accuracy_score(y_test, custom_model.predict(X_test)))

plt.plot(parameters, accuracies_val, color='yellow')
plt.plot(parameters, accuracies_train, color='green')
plt.plot(parameters, accuracies_test, color='red')
plt.savefig('KNN.png')

plt.cla()
plt.clf()

plt.plot(parameters, accuracies_val_custom, color='yellow')
plt.plot(parameters, accuracies_train_custom, color='green')
plt.plot(parameters, accuracies_test_custom, color='red')
plt.savefig('KNN_custom.png')


def lowess(X, y, model):
    w = []
    for i in range(len(X)):
        new_X = np.delete(X, i, axis=0)
        new_y = np.delete(y, i, axis=0)
        model.fit(new_X, new_y)
        y_pred = model.predict(np.array([X[i]]))[0]
        w.append(1 if y[i] != y_pred else 0)

    return w


knn_custom = KNNClassifier(n=5)

knn_custom.fit(X_train, y_train)
print(accuracy_score(y_test, knn_custom.predict(X_test)))

low = lowess(X_train.tolist(), y_train.tolist(), knn_custom)

sampled_X_train = np.array(list(map(lambda t: t[1], filter(lambda t: low[t[0]] > 0, enumerate(X_train.tolist())))))
sampled_y_train = np.array(list(map(lambda t: t[1], filter(lambda t: low[t[0]] > 0, enumerate(y_train.tolist())))))

knn_custom.fit(sampled_X_train, sampled_y_train)
print(accuracy_score(y_test, knn_custom.predict(X_test)))

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
print(accuracy_score(y_test, knn.predict(X_test)))

knn.fit(sampled_X_train, sampled_y_train)
print(accuracy_score(y_test, knn.predict(X_test)))
