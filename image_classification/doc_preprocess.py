import numpy as np
from sklearn.datasets import load_svmlight_files


def load_data(f_train, f_test):
    data = load_svmlight_files([f_train])
    X, y = data[0].astype(np.float32), data[1].astype(np.int32)

    label_map = {}
    new_label = 0

    size = np.size(y)

    for i in range(size):
        if y[i] in label_map.keys():
            y[i] = label_map[y[i]]
        else:
            label_map[y[i]] = new_label
            y[i] = new_label
            new_label += 1

    test_data = load_svmlight_files([f_test])
    test_X, test_y = test_data[0].astype(np.float32), test_data[1].astype(np.int32)

    row, actual_row = 0, 0
    while row < np.size(test_y):
        if test_y[row] in label_map.keys():
            test_X[row] = test_X[actual_row]
            test_y[row] = label_map[test_y[row]]
            row += 1
        else:
            test_y = np.delete(test_y, row)
        actual_row += 1

    return (X, y), (test_X[:row], test_y)
