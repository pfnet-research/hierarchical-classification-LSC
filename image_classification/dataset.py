import numpy as np
import random
from chainer.datasets import TupleDataset


class Dataset(TupleDataset):
    def __init__(self, instances, labels, assignment):
        clusters, classes = [assignment[label][0] for label in labels], \
                            [assignment[label][1] for label in labels]
        super(Dataset, self).__init__(instances, clusters, classes)

