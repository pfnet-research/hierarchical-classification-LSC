import numpy as np
from sklearn.datasets import load_svmlight_files


def load_data(f_train, f_test):
    """
    with open(f_train_instance, 'rb') as f:
        X = np.load(f).astype(np.float32)
    with open(f_train_label, 'rb') as f:
        y = np.load(f).astype(np.int32)
    """
    train_data = load_svmlight_files([f_train])
    test_data = load_svmlight_files([f_test])

    X, y = train_data[0], train_data[1].astype(np.int32)
    test_X, test_y = test_data[0], test_data[1].astype(np.int32)

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
    """
    with open(f_test_instance, 'rb') as f:
        test_X = np.load(f).astype(np.float32)
    with open(f_test_label, 'rb') as f:
        test_y = np.load(f).astype(np.int32)
    """

    row, actual_row = 0, 0
    while row < np.size(test_y):
        if test_y[row] in label_map.keys():
            test_X[row] = test_X[actual_row]
            test_y[row] = label_map[test_y[row]]
            row += 1
        else:
            test_y = np.delete(test_y, row)
        actual_row += 1

    return (X, y), (test_X[:row], test_y), new_label
