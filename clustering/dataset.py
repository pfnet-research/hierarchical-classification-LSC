import numpy as np
from scipy.sparse import csr_matrix
import random


class Dataset(object):
    def __init__(self, instances, labels, sparse=False):
        print(instances.shape, labels.shape)
        instances, labels = instances[np.argsort(labels)], np.sort(labels)
        label_type = np.unique(labels)
        partition = [np.searchsorted(labels, k, side='left') for k in label_type]
        partition.append(np.size(instances, axis=0))

        self._instances = instances
        self._labels = labels
        self._length = len(labels)
        self._partition = partition
        self.sparse = sparse

    def __getitem__(self, index):
        if isinstance(index, slice):
            instances = self._instances[index]

            if self.sparse:
                instances = np.array(csr_matrix.todense(instances))

            labels = self._labels[index]

            length = len(instances)

            # バッチ内の各ラベルからランダムに要素を取り出す。
            sampled_instances = [self._instances[random.choice(range(self._partition[label],
                                                                     self._partition[label + 1]))]
                                 for label in labels]
            if self.sparse:
                sampled_instances = [np.array(csr_matrix.todense(sampled_instance)) for sampled_instance in sampled_instances]

            return [(instances[i], labels[i], sampled_instances[i]) for i in range(length)]
        else:
            instance = self._instances[index]
            if self.sparse:
                instance = np.array(csr_matrix.todense(instance))

            label = self._labels[index]
            sampled_instance = self._instances[random.choice(range(self._partition[label], self._partition[label + 1]))]

            if self.sparse:
                sampled_instance = np.array(csr_matrix.todense(sampled_instance))

            return instance, label, sampled_instance

    def __len__(self):
        return self._length