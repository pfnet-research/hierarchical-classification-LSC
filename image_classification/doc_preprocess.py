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

    size = np.size(test_y)
    for i in range(size):
        if test_y[i] in label_map.keys():
            test_y[i] = label_map[test_y[i]]
        else:
            print(test_y[i])
            label_map[test_y[i]] = new_label
            test_y[i] = new_label
            new_label += 1

    return (X, y), (test_X, test_y)
