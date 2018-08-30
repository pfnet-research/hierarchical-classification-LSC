import numpy as np
from sklearn.datasets import load_svmlight_files


def load_data(f_name):
    data = load_svmlight_files([f_name])
    X, y = data[0].astype(np.float32), data[1].astype(np.int32)

    size = np.size(y)
    index = np.argsort(y)
    X = X[index]
    y = y[index]

    new_label = 0
    new_y = np.zeros(size).astype(np.int32)
    for i in range(1, size):
        if y[i - 1] == y[i]:
            new_y[i] = new_label
        else:
            new_label += 1
            new_y[i] = new_label

    return X, new_y
