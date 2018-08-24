import chainer
import chainer.functions as F
from chainer.backends import cuda
import numpy as np


def det_cluster(model, train, num_classes, batchsize=128, device=-1):
    with chainer.using_config('train', False):
        i, N = 0, len(train)
        res = None

        while i <= N:
            xx = -F.log(F.softmax(model(
                chainer.dataset.convert.concat_examples(train[i:i + batchsize], device=device)[0]))).data
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
            cluster = np.argmin(np.sum(res[partition[i]:partition[i+1]], axis=0))
            cluster_label.append(cluster)

    return cluster_label


def assign(cluster_label, num_classes, num_clusters):
    count_classes = [0 for _ in range(num_clusters)]
    res = []
    for i in range(num_classes):
        cluster = cluster_label[i]
        res.append((cluster, count_classes[cluster]))
        count_classes[cluster] += 1

    return res, count_classes
