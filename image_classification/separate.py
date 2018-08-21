import chainer
import chainer.functions as F
from chainer.backends import cuda
import numpy as np


def det_cluster(model, train, num_classes, batchsize=128, device=-1):
    i, N = 0, len(train)
    res = None

    while i <= N:
        xx = F.softmax(model(chainer.dataset.convert.concat_examples(train[i:i + batchsize], device=device)[0])).data
        if device >= 0:
            xx = cuda.to_cpu(xx)

        if res is None:
            res = xx
        else:
            res = np.append(res, xx, axis=0)
        i += batchsize

    partition = train._partition

    cluster_label = []
    for i in range(num_classes):
        cluster = np.argmax(np.sum(res[partition[i]:partition[i+1]], axis=0))
        cluster_label.append(cluster)

    return cluster_label
