from chainer.backends import cuda
import chainer.functions as F
from chainer import reporter
import numpy as np
import chainer


class Accuracy(chainer.Chain):
    def __init__(self, h_net, assignment):
        super(self.__class__, self).__init__()
        with self.init_scope():
            self.model = h_net
            self.assignment = assignment

    def __call__(self, instances, clusters, classes):
        """
        xp = cuda.get_array_module(*instances)
        index = xp.argsort(clusters)
        instances, clusters, classes = xp.take(instances, index), xp.take(clusters, index), xp.take(clusters, index)

        if xp == np:
            clusters_cpu = clusters
        else:
            clusters_cpu = cuda.to_cpu(clusters)
        partition = np.zeros(self.num_clusters).astype(np.int32)
        for cluster in clusters_cpu:
            partition[cluster] += 1
        for i in range(1, self.num_clusters):
            partition[i] += partition[i - 1]

        y = self.model.train(instances, clusters, classes, partition)
        loss = self.mle_loss(y)

        reporter.report({'loss': loss}, self)
        """
        xp = cuda.get_array_module(*instances)
        with chainer.using_config('train', False):
            cluster_y, class_y = self.model.inference(instances)
            res = xp.logical_and((clusters == cluster_y), (classes == class_y))
            acc_num = xp.sum(res)
            total = len(res)
            accuracy = acc_num / total
            reporter.report({'accuracy': accuracy}, self)

    @staticmethod
    def mle_loss(p):
        return -F.sum(F.log(p))
